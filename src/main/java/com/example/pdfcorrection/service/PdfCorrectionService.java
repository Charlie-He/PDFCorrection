package com.example.pdfcorrection.service;

import com.example.pdfcorrection.model.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.*;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import javax.imageio.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * PDF倾斜检测与矫正服务

 * 核心功能：
 *   使用Hough变换和Radon变换进行倾斜角度检测
 *   支持批量处理和并行处理
 *   自适应DPI渲染
 *   智能角度融合算法

 * 技术特点：
 *   Radon变换使用相对标准差(CV)作为score，避免量级失控
 *   精细两阶段搜索：粗搜索 + 精搜索
 *   优化融合策略，Hough优先级更高
 *   添加异常检测和回退机制
 *
 * @author PDF Correction Team
 * @version 2.0
 */
@Service
@Slf4j
public class PdfCorrectionService {

    // ==================== 配置参数 ====================

    @Value("${file.upload-dir:uploads}")
    private String uploadDir;

    @Value("${pdf.correction.dpi:200}")
    private int renderDpi;

    @Value("${pdf.correction.min-angle}")
    private double minCorrectionAngle;

    @Value("${pdf.correction.compression.quality:0.92}")
    private float compressionQuality;

    @Value("${pdf.correction.batch-size:10}")
    private int batchSize;

    @Value("${pdf.correction.use-png:false}")
    private boolean usePngFormat;

    @Value("${pdf.correction.preserve-content:false}")
    private boolean preserveContent;

    // ==================== 常量定义 ====================

    private static final int MAX_DETECTION_SIZE = 1000;
    private static final AtomicInteger POOL_NUMBER = new AtomicInteger(1);

    // 角度检测参数
    private static final double ANGLE_RANGE = 5.0;
    private static final double COARSE_STEP = 0.2;
    private static final double FINE_STEP = 0.01;

    // Hough检测参数
    private static final double BUCKET_SIZE_HOUGH = 0.02;
    private static final double AVG_RANGE_HOUGH = 0.15;
    private static final double ANGLE_FILTER_THRESHOLD = 5.0;

    // Radon检测参数
    private static final double BUCKET_SIZE_RADON = 0.2;
    private static final double MIN_SCORE_THRESHOLD = 0.02;
    private static final double RADON_SCORE_MULTIPLIER = 5.0;

    // ==================== 依赖组件 ====================

    private final ExecutorService executorService;
    private Path uploadPath;

    @Autowired
    private ProgressService progressService;

    // ==================== 构造与初始化 ====================

    /**
     * 构造函数：初始化线程池
     */
    public PdfCorrectionService() {
        this.executorService = createThreadPool();
        log.info("PDF矫正服务线程池已创建");
    }

    /**
     * 创建自定义线程池
     */
    private ExecutorService createThreadPool() {
        int corePoolSize = Runtime.getRuntime().availableProcessors();
        BlockingQueue<Runnable> workQueue = new ArrayBlockingQueue<>(8);

        ThreadFactory threadFactory = r -> {
            Thread t = new Thread(r,
                    "pdf-correction-pool-" + POOL_NUMBER.getAndIncrement() +
                            "-thread-" + new AtomicInteger(1).getAndIncrement());
            t.setDaemon(false);
            t.setPriority(Thread.NORM_PRIORITY);
            return t;
        };

        RejectedExecutionHandler handler = new ThreadPoolExecutor.CallerRunsPolicy();

        return new ThreadPoolExecutor(
                corePoolSize,
                corePoolSize,
                60L,
                TimeUnit.SECONDS,
                workQueue,
                threadFactory,
                handler
        );
    }

    /**
     * 服务初始化
     */
    @PostConstruct
    public void init() {
        nu.pattern.OpenCV.loadLocally();
        log.info("OpenCV库加载成功");

        uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
        try {
            Files.createDirectories(uploadPath);
            log.info("上传目录创建成功: {}", uploadPath);
        } catch (Exception e) {
            throw new RuntimeException("无法创建上传目录: " + uploadPath, e);
        }

        log.info("PDF矫正服务初始化完成 - 改进的 Hough + Radon 模式");
    }

    /**
     * 服务销毁
     */
    @PreDestroy
    public void shutdownExecutorService() {
        log.info("正在关闭线程池...");
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                log.warn("线程池未能在10秒内关闭，强制关闭");
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            log.error("线程池关闭被中断", e);
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
        log.info("PDF矫正服务已关闭");
    }

    // ==================== 主流程：PDF倾斜矫正 ====================

    /**
     * 执行PDF倾斜检测与矫正
     *
     * @param file 上传的PDF文件
     * @return 矫正结果，包含输出文件名、所有角度和处理时间
     * @throws Exception 处理过程中的异常
     */
    public CorrectionResult correctPdfSkewWithAngle(MultipartFile file) throws Exception {
        log.info("========== 开始处理PDF文件 ==========");
        log.info("文件名: {}", file.getOriginalFilename());
        log.info("文件大小: {} KB", file.getSize() / 1024);

        Path tempInputPath = null;
        PDDocument document = null;
        PDDocument correctedDoc = null;
        long startTime = System.currentTimeMillis();

        try {
            // 1. 文件准备
            FilePreparation prep = prepareFiles(file);
            tempInputPath = prep.tempInputPath;
            Path outputPath = prep.outputPath;

            // 2. 加载PDF文档
            document = PDDocument.load(tempInputPath.toFile());
            correctedDoc = new PDDocument();
            int numberOfPages = document.getNumberOfPages();

            log.info("PDF总页数: {}", numberOfPages);
            progressService.sendProgress("开始处理 " + numberOfPages + " 页PDF...");

            // 3. 批量处理所有页面
            List<Double> allAngles = processPdfInBatches(
                    document, correctedDoc, numberOfPages);

            // 4. 保存结果
            progressService.sendProgress("正在保存文件...");
            correctedDoc.save(outputPath.toFile());

            long totalTime = System.currentTimeMillis() - startTime;
            log.info("========== 处理完成 ==========");
            log.info("总耗时: {}秒", String.format("%.2f", totalTime / 1000.0));
            log.info("输出文件: {}", outputPath.getFileName());

            progressService.sendProgress("处理完成");
            progressService.sendProgress(String.format("总用时: %.2fs", totalTime / 1000.0));

            // 5. 发送平均角度信息
            if (!allAngles.isEmpty()) {
                double avgAngle = allAngles.stream()
                        .mapToDouble(a -> a)
                        .average()
                        .orElse(0.0);
                log.info("检测到的平均倾斜角度: {}°", String.format("%.2f", avgAngle));
                progressService.sendAngleDetected(avgAngle);
            }

            return new CorrectionResult(
                    outputPath.getFileName().toString(),
                    allAngles,
                    totalTime
            );

        } finally {
            cleanupResources(document, correctedDoc, tempInputPath);
        }
    }

    /**
     * 准备输入输出文件
     */
    private FilePreparation prepareFiles(MultipartFile file) throws Exception {
        String originalFileName = file.getOriginalFilename();
        String baseName = extractBaseName(originalFileName);
        String correctedFileName = baseName + "_corrected_" + UUID.randomUUID() + ".pdf";

        Path tempInputPath = uploadPath.resolve("temp_input_" + UUID.randomUUID() + ".pdf");
        Path outputPath = uploadPath.resolve(correctedFileName);

        file.transferTo(tempInputPath.toFile());
        log.debug("临时文件已创建: {}", tempInputPath);

        return new FilePreparation(tempInputPath, outputPath);
    }

    /**
     * 分批处理PDF页面
     */
    private List<Double> processPdfInBatches(
            PDDocument sourceDoc,
            PDDocument targetDoc,
            int numberOfPages) throws Exception {

        List<Double> allAngles = new ArrayList<>();
        int totalBatches = (int) Math.ceil((double) numberOfPages / batchSize);

        log.info("分批处理: {} 个批次, 每批 {} 页", totalBatches, batchSize);
        progressService.sendProgress("开始处理,总共 " + totalBatches + " 个批次");

        for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
            processSingleBatch(
                    sourceDoc, targetDoc, batchIndex, totalBatches,
                    numberOfPages, allAngles
            );

            // 批次间垃圾回收
            System.gc();
        }

        return allAngles;
    }

    /**
     * 处理单个批次
     */
    private void processSingleBatch(
            PDDocument sourceDoc,
            PDDocument targetDoc,
            int batchIndex,
            int totalBatches,
            int numberOfPages,
            List<Double> allAngles) throws Exception {

        int startPage = batchIndex * batchSize;
        int endPage = Math.min(startPage + batchSize, numberOfPages);

        log.info(">>> 批次 {}/{} (页面 {}-{})",
                batchIndex + 1, totalBatches, startPage + 1, endPage);
        long batchStartTime = System.currentTimeMillis();

        // 阶段1: 渲染页面
        long renderStart = System.currentTimeMillis();
        List<PageData> batchPageData = renderPagesBatch(sourceDoc, startPage, endPage);
        logPhaseTime("渲染", renderStart, batchIndex, totalBatches, startPage, endPage);

        // 阶段2: 角度检测
        long detectionStart = System.currentTimeMillis();
        List<PageAngleResult> batchAngles = detectSkewAnglesBatch(batchPageData);
        logPhaseTime("角度检测", detectionStart, batchIndex, totalBatches, startPage, endPage);

        // 阶段3: 图像矫正
        long correctionStart = System.currentTimeMillis();
        List<CorrectedPageData> correctedPages = correctImagesBatchOptimized(
                batchPageData, batchAngles);
        logPhaseTime("图像矫正", correctionStart, batchIndex, totalBatches, startPage, endPage);

        // 阶段4: 写入PDF
        long writeStart = System.currentTimeMillis();
        writeCorrectedPagesToPdfOptimized(targetDoc, sourceDoc, correctedPages);
        logPhaseTime("写入PDF", writeStart, batchIndex, totalBatches, startPage, endPage);

        long batchTotal = System.currentTimeMillis() - batchStartTime;
        log.info("批次 {}/{} 总用时: {}ms", batchIndex + 1, totalBatches, batchTotal);
        progressService.sendProgress(String.format(
                "批次 %d/%d (页面 %d-%d) 总用时: %dms",
                batchIndex + 1, totalBatches, startPage + 1, endPage, batchTotal));

        // 收集角度信息
        allAngles.addAll(batchAngles.stream()
                .map(PageAngleResult::getAngle)
                .toList());

        // 清理批次资源
        cleanupBatchResources(batchPageData, correctedPages);
    }

    /**
     * 记录处理阶段耗时
     */
    private void logPhaseTime(
            String phaseName,
            long startTime,
            int batchIndex,
            int totalBatches,
            int startPage,
            int endPage) {

        long elapsed = System.currentTimeMillis() - startTime;
        log.debug("  - {} 完成: {}ms", phaseName, elapsed);
        progressService.sendProgress(String.format(
                "批次 %d/%d (页面 %d-%d) %s完成,用时: %dms",
                batchIndex + 1, totalBatches, startPage + 1, endPage, phaseName, elapsed));
    }

    // ==================== 页面渲染 ====================

    /**
     * 多线程并行渲染PDF页面
     *
     * @param document PDF文档
     * @param startPage 起始页码
     * @param endPage 结束页码
     * @return 渲染后的页面数据列表
     */
    private List<PageData> renderPagesBatch(
            PDDocument document,
            int startPage,
            int endPage) {

        log.debug("开始渲染页面 {}-{}", startPage + 1, endPage);
        ConcurrentHashMap<Integer, PageData> resultMap = new ConcurrentHashMap<>();
        List<Future<?>> futures = new ArrayList<>();

        for (int pageIndex = startPage; pageIndex < endPage; pageIndex++) {
            final int currentPage = pageIndex;
            futures.add(executorService.submit(() ->
                    renderSinglePage(document, currentPage, resultMap)));
        }

        // 等待所有任务完成
        waitForFutures(futures);

        // 按页码排序返回
        return resultMap.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(Map.Entry::getValue)
                .toList();
    }

    /**
     * 渲染单个页面
     */
    private void renderSinglePage(
            PDDocument document,
            int pageIndex,
            ConcurrentHashMap<Integer, PageData> resultMap) {

        try {
            PDFRenderer renderer = new PDFRenderer(document);
            renderer.setSubsamplingAllowed(false);

            PDPage page = document.getPage(pageIndex);
            float widthPt = page.getMediaBox().getWidth();
            int adaptiveDpi = adaptDpi(widthPt);

            BufferedImage image = renderer.renderImageWithDPI(
                    pageIndex,
                    adaptiveDpi,
                    ImageType.RGB
            );

            resultMap.put(pageIndex, new PageData(
                    pageIndex, image, page.getMediaBox()));

            log.trace("页面 {} 渲染完成 (DPI: {})", pageIndex + 1, adaptiveDpi);

        } catch (Exception e) {
            log.error("渲染第 {} 页失败: {}", pageIndex + 1, e.getMessage(), e);
        }
    }

    /**
     * 根据页面宽度动态调整渲染DPI
     *
     * @param widthPt 页面宽度(点)
     * @return 调整后的DPI值
     */
    private int adaptDpi(float widthPt) {
        if (widthPt > 800) {
            return Math.min(renderDpi, 180); // A3或更大
        }
        if (widthPt < 400) {
            return Math.max(renderDpi, 220); // 小页略提DPI
        }
        return renderDpi;
    }

    // ==================== 倾斜角度检测(核心算法) ====================

    /**
     * 批量检测页面倾斜角度
     */
    private List<PageAngleResult> detectSkewAnglesBatch(List<PageData> pageDataList) {
        log.debug("开始批量角度检测, 页面数: {}", pageDataList.size());

        List<CompletableFuture<PageAngleResult>> futures = pageDataList.stream()
                .map(pd -> CompletableFuture.supplyAsync(
                        () -> detectPageSkewAngle(pd), executorService))
                .toList();

        return futures.stream()
                .map(CompletableFuture::join)
                .collect(Collectors.toList());
    }

    /**
     * 单页倾斜角度检测：融合Hough和Radon算法
     *
     * <p>检测流程：
     * <ol>
     *   <li>预处理：转灰度图、降噪</li>
     *   <li>Hough变换检测</li>
     *   <li>Radon变换检测</li>
     *   <li>融合两种算法的结果</li>
     *   <li>小角度过滤</li>
     * </ol>
     *
     * @param pageData 页面数据
     * @return 检测到的角度结果
     */
    public PageAngleResult detectPageSkewAngle(PageData pageData) {
        Mat original = null;
        Mat gray = null;
        Mat processed = null;

        try {
            // 1. 图像预处理
            original = bufferedImageToMat(pageData.getImage());
            gray = preprocessForScannedDoc(original);

            // 2. 自适应缩放
            double scale = calculateScaleFactor(gray);
            processed = new Mat();
            if (scale < 1.0) {
                Imgproc.resize(gray, processed, new Size(),
                        scale, scale, Imgproc.INTER_AREA);
            } else {
                processed = gray.clone();
            }

            // 3. Hough变换检测
            AngleScore houghAS = detectByProbabilisticHoughWithScore(processed);

            // 4. Radon变换检测
            AngleScore radonAS = detectByRadonOnTextMask(processed);

            // 5. 融合角度
            double finalAngle = fuseTwoAngles(houghAS, radonAS);

            // 6. 小角度过滤
            if (Math.abs(finalAngle) < minCorrectionAngle) {
                finalAngle = 0.0;
            }

            log.info("第 {} 页 | Hough: {}° (score={}) | Radon: {}° (score={}) | 最终: {}°",
                    pageData.getPageIndex() + 1,
                    String.format("%.2f", houghAS.angle),
                    String.format("%.0f", houghAS.score),
                    String.format("%.2f", radonAS.angle),
                    String.format("%.3f", radonAS.score),
                    String.format("%.2f", finalAngle));

            return new PageAngleResult(pageData.getPageIndex(), finalAngle);

        } catch (Exception e) {
            log.error("检测第 {} 页角度失败", pageData.getPageIndex() + 1, e);
            return new PageAngleResult(pageData.getPageIndex(), 0.0);
        } finally {
            releaseMat(original, gray, processed);
        }
    }

    // ==================== Hough变换检测 ====================

    /**
     * 基于概率Hough变换的倾斜检测
     *
     * <p>改进点：
     * <ul>
     *   <li>提高theta分辨率至1°/1080</li>
     *   <li>严格过滤：仅接受[-5°, 5°]范围</li>
     *   <li>使用长度平方作为权重</li>
     *   <li>细化桶大小至0.02°</li>
     * </ul>
     *
     * @param gray 灰度图像
     * @return 角度和置信度
     */
    private AngleScore detectByProbabilisticHoughWithScore(Mat gray) {
        Mat binary = null;
        Mat horizontal = null;
        Mat lines = null;

        try {
            // 1. 二值化
            binary = new Mat();
            Imgproc.threshold(gray, binary, 0, 255,
                    Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // 2. 形态学处理：增强文本行连贯性
            horizontal = new Mat();
            int kernelWidth = Math.max(20, gray.cols() / 60);
            Mat kernelH = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT, new Size(kernelWidth, 1));
            Imgproc.dilate(binary, horizontal, kernelH);
            kernelH.release();

            // 3. Canny边缘检测
            Imgproc.Canny(horizontal, horizontal, 50, 150, 3, false);

            // 4. 配置HoughLinesP参数
            int minLineLength = Math.max(gray.cols() / 6, 100);
            int maxLineGap = Math.max(gray.cols() / 15, 30);
            int threshold = 50;

            // 5. 执行Hough变换
            lines = new Mat();
            Imgproc.HoughLinesP(horizontal, lines, 1, Math.PI / 1080,
                    threshold, minLineLength, maxLineGap);

            if (lines.rows() < 2) {
                log.debug("Hough检测到的线段数量不足");
                return new AngleScore(Double.NaN, 0.0);
            }

            // 6. 计算加权角度分布
            Map<Double, Double> bucketWeight = new HashMap<>();

            for (int i = 0; i < lines.rows(); i++) {
                double[] l = lines.get(i, 0);
                double dx = l[2] - l[0];
                double dy = l[3] - l[1];
                double len = Math.hypot(dx, dy);

                double angle = Math.toDegrees(Math.atan2(dy, dx));
                angle = normalizeAngle(angle);

                // 过滤：仅接受接近水平的线段
                if (Math.abs(angle) > ANGLE_FILTER_THRESHOLD) {
                    continue;
                }

                double bucket = Math.round(angle / BUCKET_SIZE_HOUGH) * BUCKET_SIZE_HOUGH;
                bucketWeight.merge(bucket, len * len, Double::sum);
            }

            if (bucketWeight.isEmpty()) {
                log.debug("Hough: 没有有效的水平线段");
                return new AngleScore(Double.NaN, 0.0);
            }

            // 7. 找到得分最高的桶
            double bestBucket = bucketWeight.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .get()
                    .getKey();

            // 8. 在最佳桶附近进行加权平均
            double weightedSum = 0.0;
            double totalWeight = 0.0;

            for (Map.Entry<Double, Double> entry : bucketWeight.entrySet()) {
                double bucket = entry.getKey();
                double weight = entry.getValue();

                if (Math.abs(bucket - bestBucket) <= AVG_RANGE_HOUGH) {
                    weightedSum += bucket * weight;
                    totalWeight += weight;
                }
            }

            double avgAngle = (totalWeight > 0) ?
                    (weightedSum / totalWeight) : bestBucket;
            double score = bucketWeight.getOrDefault(bestBucket, 0.0);

            log.debug("Hough检测完成: angle={}°, score={}", 
                    String.format("%.2f", avgAngle), 
                    String.format("%.0f", score));
            return new AngleScore(avgAngle, score);

        } catch (Exception e) {
            log.error("Hough检测异常", e);
            return new AngleScore(Double.NaN, 0.0);
        } finally {
            releaseMat(binary, horizontal, lines);
        }
    }

    // ==================== Radon变换检测 ====================

    /**
     * 基于文本掩码的Radon变换检测
     *
     * <p>优化策略：
     * <ul>
     *   <li>创建文本掩码过滤非文本区域</li>
     *   <li>移除上采样以降低计算量</li>
     *   <li>使用块检测增强鲁棒性</li>
     * </ul>
     */
    private AngleScore detectByRadonOnTextMask(Mat gray) {
        Mat textMask = null;
        Mat textOnly = null;
        Mat up = null;

        try {
            // 1. 创建文本掩码
            textMask = createTextMask(gray);
            if (textMask == null ||
                    Core.countNonZero(textMask) < (textMask.total() * 0.001)) {
                log.debug("文本区域不足，使用全图Radon");
                return detectByFullRadon(gray);
            }

            // 2. 提取文本区域
            textOnly = new Mat();
            gray.copyTo(textOnly, textMask);

            if (textOnly.empty() || Core.countNonZero(textOnly) == 0) {
                log.debug("文本提取失败，使用块检测");
                return detectByBlockwiseRadon(gray);
            }

            // 3. 直接使用提取的文本区域(移除上采样)
            up = textOnly.clone();

            // 4. 块检测
            AngleScore blockAS = detectByBlockwiseRadon(up);

            // 5. 置信度检查
            if (blockAS.score < 0.5) {
                log.debug("块检测置信度低({})，补充全图检测", String.format("%.3f", blockAS.score));
                AngleScore fullAS = detectByFullRadon(gray);
                if (fullAS.score > blockAS.score) {
                    return fullAS;
                }
            }

            return blockAS;

        } catch (Exception e) {
            log.error("Radon (TextMask) 检测异常", e);
            return detectByBlockwiseRadon(gray);
        } finally {
            releaseMat(textMask, textOnly, up);
        }
    }

    /**
     * 全图高精度Radon检测
     */
    private AngleScore detectByFullRadon(Mat gray) {
        Mat up = null;
        try {
            double upScale = 2.0;
            up = new Mat();
            Imgproc.resize(gray, up, new Size(),
                    upScale, upScale, Imgproc.INTER_CUBIC);

            return detectRadonBlock(up);
        } finally {
            releaseMat(up);
        }
    }

    /**
     * 创建文本掩码
     *
     * <p>过滤策略：
     * <ul>
     *   <li>基于轮廓的面积、宽高比筛选</li>
     *   <li>动态调整参数适应不同DPI</li>
     *   <li>过滤大块图形和细线条</li>
     * </ul>
     *
     * @param gray 灰度图
     * @return 文本区域掩码(255=文本, 0=背景)
     */
    private Mat createTextMask(Mat gray) {
        Mat binary = null;
        Mat hierarchy = null;
        List<MatOfPoint> contours = new ArrayList<>();
        Mat textMask = null;

        try {
            // 1. 自适应二值化
            binary = new Mat();
            Imgproc.adaptiveThreshold(gray, binary, 255,
                    Imgproc.ADAPTIVE_THRESH_MEAN_C,
                    Imgproc.THRESH_BINARY, 21, 5);

            // 2. 查找所有轮廓
            hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy,
                    Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            // 3. 准备掩码画布
            textMask = Mat.zeros(gray.size(), CvType.CV_8UC1);

            // 4. 动态计算过滤参数
            double scale = calculateScaleFactor(gray);
            FilterParams params = calculateFilterParams(scale);

            log.trace("文本掩码参数: minH={}, maxH={}, minW={}, maxW={}, minArea={}, maxArea={}",
                    params.minHeight, params.maxHeight, params.minWidth,
                    params.maxWidth, params.minArea, params.maxArea);

            // 5. 过滤并提取文本轮廓
            List<MatOfPoint> textContours = new ArrayList<>();
            for (MatOfPoint contour : contours) {
                if (isTextContour(contour, params)) {
                    textContours.add(contour);
                }
            }

            log.debug("文本轮廓筛选: 总数={}, 文本={}",
                    contours.size(), textContours.size());

            // 6. 绘制文本掩码
            Imgproc.drawContours(textMask, textContours, -1,
                    new Scalar(255), -1);

            // 7. 形态学闭运算：连接断裂字符
            Mat kernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT, new Size(5, 5));
            Imgproc.morphologyEx(textMask, textMask,
                    Imgproc.MORPH_CLOSE, kernel);
            kernel.release();

            return textMask;

        } finally {
            releaseMat(binary, hierarchy);
            contours.forEach(MatOfPoint::release);
        }
    }

    /**
     * 计算文本过滤参数
     */
    private FilterParams calculateFilterParams(double scale) {
        FilterParams params = new FilterParams();
        params.minHeight = (int) Math.max(8, 10 * scale);
        params.maxHeight = (int) Math.max(100, 150 * scale);
        params.minWidth = (int) Math.max(2, 3 * scale);
        params.maxWidth = (int) Math.max(100, 150 * scale);
        params.minArea = Math.max(15, 20 * scale);
        params.maxArea = Math.max(2500, 3000 * scale);
        params.minAspect = 0.08;
        params.maxAspect = 4.0;
        return params;
    }

    /**
     * 判断是否为文本轮廓
     */
    private boolean isTextContour(MatOfPoint contour, FilterParams params) {
        double area = Imgproc.contourArea(contour);

        // 面积过滤
        if (area < params.minArea || area > params.maxArea) {
            return false;
        }

        Rect rect = Imgproc.boundingRect(contour);

        // 高度过滤
        if (rect.height < params.minHeight || rect.height > params.maxHeight) {
            return false;
        }

        // 宽度过滤
        if (rect.width < params.minWidth || rect.width > params.maxWidth) {
            return false;
        }

        // 宽高比过滤
        double aspectRatio = (double) rect.width / (double) rect.height;
        return !(aspectRatio < params.minAspect) && !(aspectRatio > params.maxAspect);
    }

    /**
     * 块级Radon检测
     *
     * <p>使用加权投票直方图代替中位数，让高置信度块有更大话语权
     */
    private AngleScore detectByBlockwiseRadon(Mat gray) {
        int blockSize = Math.max(256, Math.min(gray.cols(), gray.rows()) / 8);
        int step = blockSize / 2;
        int rows = gray.rows();
        int cols = gray.cols();

        log.debug("块级Radon检测: blockSize={}, step={}", blockSize, step);
        List<AngleScore> results = new ArrayList<>();

        // 1. 收集所有有效块的AngleScore
        if (rows <= blockSize || cols <= blockSize) {
            results.add(detectRadonBlock(gray));
        } else {
            for (int y = 0; y + blockSize <= rows; y += step) {
                for (int x = 0; x + blockSize <= cols; x += step) {
                    Rect roi = new Rect(x, y, blockSize, blockSize);
                    Mat block = new Mat(gray, roi);
                    AngleScore as = detectRadonBlock(block);
                    block.release();

                    if (!Double.isNaN(as.angle) && as.score > MIN_SCORE_THRESHOLD) {
                        results.add(as);
                    }
                }
            }
        }

        if (results.isEmpty()) {
            log.debug("没有有效的Radon块结果，使用全图检测");
            return detectRadonBlock(gray);
        }

        log.debug("有效Radon块数量: {}", results.size());

        // 2. 加权投票直方图
        Map<Double, Double> bucketWeight = new HashMap<>();

        for (AngleScore as : results) {
            double bucket = Math.round(as.angle / BUCKET_SIZE_RADON) * BUCKET_SIZE_RADON;
            bucketWeight.merge(bucket, as.score, Double::sum);
        }

        if (bucketWeight.isEmpty()) {
            return new AngleScore(Double.NaN, 0.0);
        }

        // 3. 找到得分最高的桶
        double bestBucket = bucketWeight.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get()
                .getKey();

        // 4. 在最佳桶附近进行加权平均
        double weightedSum = 0.0;
        double totalWeight = 0.0;
        double scoreSum = 0.0;
        double count = 0.0;

        for (AngleScore as : results) {
            double bucket = Math.round(as.angle / BUCKET_SIZE_RADON) * BUCKET_SIZE_RADON;
            if (Math.abs(bucket - bestBucket) <= 0.5) {
                weightedSum += as.angle * as.score;
                totalWeight += as.score;
                scoreSum += as.score;
                count++;
            }
        }

        double avgAngle = (totalWeight > 0) ?
                (weightedSum / totalWeight) : bestBucket;
        double avgScore = (count > 0) ? (scoreSum / count) : 0.0;

        log.debug("块级Radon结果: angle={}°, score={}", 
                String.format("%.2f", avgAngle), 
                String.format("%.3f", avgScore));
        return new AngleScore(avgAngle, avgScore);
    }

    /**
     * 单块Radon检测(优化版)
     *
     * <p>两阶段搜索：
     * <ol>
     *   <li>粗搜索: 步长0.2°, 范围[-5°, 5°]</li>
     *   <li>精搜索: 步长0.01°, 范围[最佳角度±0.25°]</li>
     * </ol>
     */
    private AngleScore detectRadonBlock(Mat block) {
        Mat binary = null;

        try {
            // 1. 二值化
            binary = new Mat();
            Imgproc.threshold(block, binary, 0, 255,
                    Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // 2. 内容检查
            double nonZero = Core.countNonZero(binary);
            double ratio = nonZero / (double) (binary.rows() * binary.cols());

            if (ratio < 0.01 || ratio > 0.95) {
                log.trace("块内容比例异常: {}%", String.format("%.2f", ratio * 100));
                return new AngleScore(Double.NaN, 0.0);
            }

            double bestAngle = 0.0;
            double maxScore = -1.0;

            // 3. 粗搜索
            for (double a = -ANGLE_RANGE; a <= ANGLE_RANGE; a += COARSE_STEP) {
                double score = calculateRadonCV(binary, a);
                if (score > maxScore) {
                    maxScore = score;
                    bestAngle = a;
                }
            }

            // 4. 精搜索
            double fineStart = Math.max(-ANGLE_RANGE, bestAngle - 0.25);
            double fineEnd = Math.min(ANGLE_RANGE, bestAngle + 0.25);

            for (double a = fineStart; a <= fineEnd; a += FINE_STEP) {
                double score = calculateRadonCV(binary, a);
                if (score > maxScore) {
                    maxScore = score;
                    bestAngle = a;
                }
            }

            return new AngleScore(bestAngle, maxScore);

        } finally {
            releaseMat(binary);
        }
    }

    /**
     * 计算Radon变换的变异系数(CV)
     *
     * <p>使用CV²/mean作为score，避免量级失控
     */
    private double calculateRadonCV(Mat binary, double angle) {
        Mat rot = null;
        Mat proj = null;
        MatOfDouble mean = null;
        MatOfDouble std = null;

        try {
            // 1. 旋转图像
            rot = rotateImage(binary, angle);

            // 2. 垂直投影
            proj = new Mat();
            Core.reduce(rot, proj, 1, Core.REDUCE_SUM, CvType.CV_32F);

            // 3. 计算均值和标准差
            mean = new MatOfDouble();
            std = new MatOfDouble();
            Core.meanStdDev(proj, mean, std);

            double meanVal = mean.get(0, 0)[0];
            double stdVal = std.get(0, 0)[0];

            if (meanVal < 1.0) {
                return 0.0;
            }

            // 4. 返回归一化的CV²
            return (stdVal * stdVal) / (meanVal + 1e-5);

        } finally {
            releaseMat(rot, proj);
            if (mean != null) mean.release();
            if (std != null) std.release();
        }
    }

    /**
     * 角度融合算法(优化版)
     *
     * <p>融合策略：
     * <ul>
     *   <li>使用Softmax计算权重</li>
     *   <li>Radon score放大5倍平衡初始权重</li>
     *   <li>根据角度差异和绝对值调整权重</li>
     * </ul>
     *
     * @param h Hough检测结果
     * @param r Radon检测结果
     * @return 融合后的角度
     */
    private double fuseTwoAngles(AngleScore h, AngleScore r) {
        // 1. 处理NaN情况
        boolean hNaN = (h == null) || Double.isNaN(h.angle);
        boolean rNaN = (r == null) || Double.isNaN(r.angle);

        if (hNaN && rNaN) return 0.0;
        if (hNaN) return r.angle;
        if (rNaN) return h.angle;

        // 2. Score标准化
        double hScaled = Math.log1p(Math.max(0.0, h.score));
        double rScaled = Math.log1p(Math.max(0.0, r.score * RADON_SCORE_MULTIPLIER));

        if (hScaled < 1e-6 && rScaled < 1e-6) {
            return 0.0;
        }

        // 3. Softmax权重计算
        double maxv = Math.max(hScaled, rScaled);
        double hExp = Math.exp(hScaled - maxv);
        double rExp = Math.exp(rScaled - maxv);
        double wH = hExp / (hExp + rExp);
        double wR = rExp / (hExp + rExp);

        // 4. 计算角度差异
        double delta = Math.abs(h.angle - r.angle);
        double hAbs = Math.abs(h.angle);
        double rAbs = Math.abs(r.angle);
        double meanAbs = (hAbs + rAbs) / 2.0;

        // 5. 启发式规则调整权重
        if (delta < 0.2 && meanAbs > 0.1) {
            // 角度接近且非零：增强置信度
            wH = Math.sqrt(wH);
            wR = Math.sqrt(wR);
            if (Math.signum(h.angle) == Math.signum(r.angle)) {
                wH *= 1.1;
                wR *= 1.1;
            }
        } else if (delta > 0.4) {
            // 角度差异大：倾向于非零结果
            if (hAbs < 0.2 && rAbs >= 0.2) {
                wR *= 1.5;
                wH *= 0.8;
            } else if (rAbs < 0.2 && hAbs >= 0.2) {
                wH *= 1.5;
                wR *= 0.8;
            } else {
                if (hScaled > rScaled) wH *= 1.1;
                else wR *= 1.1;
            }
        }

        // 6. 归一化权重
        double sum = wH + wR;
        if (sum <= 0) {
            return (h.angle + r.angle) / 2.0;
        }
        wH /= sum;
        wR /= sum;

        double fusedAngle = h.angle * wH + r.angle * wR;
        log.debug("角度融合: H={}°(w={}) + R={}°(w={}) = {}°",
                String.format("%.2f", h.angle), String.format("%.2f", wH),
                String.format("%.2f", r.angle), String.format("%.2f", wR),
                String.format("%.2f", fusedAngle));

        return fusedAngle;
    }

    // ==================== 图像矫正与输出 ====================

    /**
     * 批量矫正图像(优化版)
     *
     * <p>优化：对不需要矫正的页面直接标记，避免重渲染
     */
    private List<CorrectedPageData> correctImagesBatchOptimized(
            List<PageData> pageDataList,
            List<PageAngleResult> angleResults) {

        log.debug("开始批量矫正图像");
        List<CorrectedPageData> corrected = new ArrayList<>();

        for (int i = 0; i < pageDataList.size(); i++) {
            PageData pd = pageDataList.get(i);
            double angle = (i < angleResults.size()) ?
                    angleResults.get(i).getAngle() : 0.0;

            boolean needsCorrection = Math.abs(angle) > minCorrectionAngle;

            if (needsCorrection) {
                // 需要矫正：渲染并旋转图像
                BufferedImage corr = correctImageSkew(pd.getImage(), angle);
                corrected.add(new CorrectedPageData(
                        pd.getPageIndex(), corr, pd.getOriginalSize(), true));
                log.debug("  第 {} 页需要矫正: {}°", pd.getPageIndex() + 1, String.format("%.2f", angle));
            } else {
                // 不需要矫正：标记为直接复制
                corrected.add(new CorrectedPageData(
                        pd.getPageIndex(), null, pd.getOriginalSize(), false));
                log.debug("  第 {} 页无需矫正", pd.getPageIndex() + 1);
            }
        }

        return corrected;
    }

    /**
     * 将矫正后的页面写入PDF(优化版)
     *
     * <p>两阶段处理：
     * <ol>
     *   <li>并行编码阶段：图像转字节数组</li>
     *   <li>顺序写入阶段：创建PDF页面</li>
     * </ol>
     */
    private void writeCorrectedPagesToPdfOptimized(
            PDDocument targetDoc,
            PDDocument sourceDoc,
            List<CorrectedPageData> correctedPages) throws Exception {

        log.debug("开始写入矫正后的页面到PDF");

        // Step 1: 并行编码阶段
        List<CompletableFuture<PageRenderResult>> futures = correctedPages.stream()
                .map(pageData -> CompletableFuture.supplyAsync(
                        () -> encodePageImage(pageData), executorService))
                .toList();

        // 等待全部完成并排序
        List<PageRenderResult> results = futures.stream()
                .map(CompletableFuture::join)
                .sorted(Comparator.comparingInt(r -> r.pageIndex))
                .toList();

        // Step 2: 顺序写入阶段
        for (PageRenderResult result : results) {
            writeSinglePage(targetDoc, sourceDoc, result);
        }

        log.info("所有 {} 页写入完成", results.size());
    }

    /**
     * 编码单个页面图像
     */
    private PageRenderResult encodePageImage(CorrectedPageData pageData) {
        try {
            if (!pageData.needsCorrection()) {
                // 无需矫正的页面，不处理图像
                return new PageRenderResult(
                        pageData.getPageIndex(), null, false, pageData.getOriginalSize());
            }

            BufferedImage image = pageData.getImage();
            PDRectangle originalSize = pageData.getOriginalSize();

            // 图像编码
            ByteArrayOutputStream baos = new ByteArrayOutputStream(1024 * 1024);

            if (usePngFormat) {
                ImageIO.write(image, "PNG", baos);
            } else {
                encodeJpeg(image, baos);
            }

            byte[] imgBytes = baos.toByteArray();
            log.trace("页面 {} 图像编码完成: {} KB",
                    pageData.getPageIndex() + 1, imgBytes.length / 1024);

            return new PageRenderResult(
                    pageData.getPageIndex(), imgBytes, true, originalSize);

        } catch (Exception e) {
            log.error("第 {} 页图像编码失败", pageData.getPageIndex() + 1, e);
            return new PageRenderResult(
                    pageData.getPageIndex(), null, false, pageData.getOriginalSize());
        }
    }

    /**
     * JPEG编码
     */
    private void encodeJpeg(BufferedImage image, ByteArrayOutputStream baos)
            throws Exception {

        ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
        ImageWriteParam params = writer.getDefaultWriteParam();
        params.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        params.setCompressionQuality(
                Math.min(Math.max(compressionQuality, 0.5f), 0.95f));

        writer.setOutput(ImageIO.createImageOutputStream(baos));
        writer.write(null, new IIOImage(image, null, null), params);
        writer.dispose();
    }

    /**
     * 写入单个页面到PDF
     */
    private void writeSinglePage(
            PDDocument targetDoc,
            PDDocument sourceDoc,
            PageRenderResult result) throws Exception {

        int pageIndex = result.pageIndex;

        if (!result.corrected || result.imageBytes == null) {
            // 无需矫正的页面：直接复制
            PDPage originalPage = sourceDoc.getPage(pageIndex);
            targetDoc.importPage(originalPage);
            log.debug("✓ 第 {} 页无需矫正，已复制", pageIndex + 1);
            return;
        }

        // 需要矫正的页面：创建新页面
        PDRectangle size = result.originalSize;
        float pdfWidth = size.getWidth();
        float pdfHeight = size.getHeight();

        PDPage pdPage = new PDPage(new PDRectangle(pdfWidth, pdfHeight));
        targetDoc.addPage(pdPage);

        PDImageXObject pdImage = PDImageXObject.createFromByteArray(
                targetDoc, result.imageBytes,
                usePngFormat ? "page.png" : "page.jpg");

        try (PDPageContentStream contentStream = new PDPageContentStream(
                targetDoc, pdPage, PDPageContentStream.AppendMode.OVERWRITE, true, true)) {
            contentStream.drawImage(pdImage, 0, 0, pdfWidth, pdfHeight);
        }

        log.trace("第 {} 页已写入 ({}×{})", pageIndex + 1, pdfWidth, pdfHeight);
    }

    /**
     * 图像预处理：适用于扫描文档
     */
    private Mat preprocessForScannedDoc(Mat original) {
        Mat gray = new Mat();

        // 转灰度
        if (original.channels() >= 3) {
            Imgproc.cvtColor(original, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = original.clone();
        }

        // 中值滤波降噪
        Imgproc.medianBlur(gray, gray, 3);

        return gray;
    }

    /**
     * 矫正图像倾斜
     *
     * @param image 原始图像
     * @param angle 旋转角度
     * @return 矫正后的图像
     */
    private BufferedImage correctImageSkew(BufferedImage image, double angle) {
        if (image == null || Math.abs(angle) < minCorrectionAngle) {
            return image;
        }

        Mat mat = null;
        Mat rotated = null;

        try {
            mat = bufferedImageToMat(image);
            Point center = new Point(mat.cols() / 2.0, mat.rows() / 2.0);
            Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);

            rotated = new Mat();

            if (preserveContent) {
                // 模式1：扩大边界保留完整内容
                rotated = rotateWithBorderExpansion(mat, rotMat, angle);
            } else {
                // 模式2：保持原始尺寸(推荐)
                Imgproc.warpAffine(mat, rotated, rotMat, mat.size(),
                        Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT,
                        new Scalar(255, 255, 255));
            }

            rotMat.release();
            return matToBufferedImage(rotated);

        } finally {
            releaseMat(mat, rotated);
        }
    }

    /**
     * 带边界扩展的旋转
     */
    private Mat rotateWithBorderExpansion(Mat mat, Mat rotMat, double angle) {
        double radians = Math.toRadians(angle);
        double sin = Math.abs(Math.sin(radians));
        double cos = Math.abs(Math.cos(radians));

        int newWidth = (int) Math.ceil(mat.width() * cos + mat.height() * sin);
        int newHeight = (int) Math.ceil(mat.width() * sin + mat.height() * cos);

        // 调整平移量
        double[] tx = rotMat.get(0, 2);
        double[] ty = rotMat.get(1, 2);
        tx[0] += (newWidth - mat.cols()) / 2.0;
        ty[0] += (newHeight - mat.rows()) / 2.0;
        rotMat.put(0, 2, tx);
        rotMat.put(1, 2, ty);

        Mat result = new Mat();
        Imgproc.warpAffine(mat, result, rotMat, new Size(newWidth, newHeight),
                Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT,
                new Scalar(255, 255, 255));

        return result;
    }

    // ==================== 辅助工具方法 ====================

    /**
     * 计算缩放因子
     */
    private double calculateScaleFactor(Mat mat) {
        final double MIN_SCALE = 0.5;
        int longEdge = Math.max(mat.cols(), mat.rows());

        if (longEdge <= MAX_DETECTION_SIZE) {
            return 1.0;
        }

        double scale = (double) MAX_DETECTION_SIZE / (double) longEdge;
        return Math.max(scale, MIN_SCALE);
    }

    /**
     * 角度归一化到[-45°, 45°]
     */
    private double normalizeAngle(double angle) {
        while (angle > 45) angle -= 90;
        while (angle < -45) angle += 90;
        return angle;
    }

    /**
     * BufferedImage转Mat
     */
    private Mat bufferedImageToMat(BufferedImage image) {
        if (image == null) return new Mat();

        BufferedImage converted = new BufferedImage(
                image.getWidth(), image.getHeight(),
                BufferedImage.TYPE_3BYTE_BGR);

        Graphics2D g = converted.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();

        byte[] pixels = ((DataBufferByte) converted.getRaster()
                .getDataBuffer()).getData();

        Mat mat = new Mat(converted.getHeight(), converted.getWidth(),
                CvType.CV_8UC3);
        mat.put(0, 0, pixels);

        return mat;
    }

    /**
     * Mat转BufferedImage
     */
    private BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) return null;

        int type = mat.channels() == 1 ?
                BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;

        BufferedImage image = new BufferedImage(
                mat.cols(), mat.rows(), type);

        byte[] data = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, data);

        byte[] target = ((DataBufferByte) image.getRaster()
                .getDataBuffer()).getData();
        System.arraycopy(data, 0, target, 0, data.length);

        return image;
    }

    /**
     * 旋转图像(适用于二值/文档图像)
     *
     * <p>特点：无裁切、保持边缘锐利
     */
    private Mat rotateImage(Mat src, double angle) {
        int w = src.cols();
        int h = src.rows();
        Point center = new Point(w / 2.0, h / 2.0);

        Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        // 计算旋转后所需的完整包围框
        double absCos = Math.abs(rotMat.get(0, 0)[0]);
        double absSin = Math.abs(rotMat.get(0, 1)[0]);
        int newW = (int) Math.round(h * absSin + w * absCos);
        int newH = (int) Math.round(h * absCos + w * absSin);

        // 调整平移量
        rotMat.put(0, 2, rotMat.get(0, 2)[0] + (newW / 2.0 - center.x));
        rotMat.put(1, 2, rotMat.get(1, 2)[0] + (newH / 2.0 - center.y));

        Mat dst = new Mat();
        int interp = Imgproc.INTER_NEAREST;
        Scalar bgColor = src.channels() == 1 ?
                new Scalar(255) : new Scalar(255, 255, 255);

        Imgproc.warpAffine(src, dst, rotMat, new Size(newW, newH),
                interp, Core.BORDER_CONSTANT, bgColor);

        rotMat.release();
        return dst;
    }

    /**
     * 释放Mat资源(可变参数版本)
     */
    private void releaseMat(Mat... mats) {
        for (Mat mat : mats) {
            if (mat != null && !mat.empty()) {
                mat.release();
            }
        }
    }

    /**
     * 清理批次资源
     */
    private void cleanupBatchResources(
            List<PageData> pageDataList,
            List<CorrectedPageData> correctedPages) {

        for (PageData pd : pageDataList) {
            if (pd.getImage() != null) {
                pd.getImage().flush();
            }
        }

        for (CorrectedPageData cp : correctedPages) {
            if (cp.getImage() != null) {
                cp.getImage().flush();
            }
        }

        log.trace("批次资源已清理");
    }

    /**
     * 清理所有资源
     */
    private void cleanupResources(
            PDDocument document,
            PDDocument correctedDoc,
            Path tempInputPath) {

        closeResource(document);
        closeResource(correctedDoc);
        deleteFile(tempInputPath);

        log.debug("所有资源已清理");
    }

    /**
     * 等待所有Future完成
     */
    private void waitForFutures(List<Future<?>> futures) {
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception e) {
                log.error("任务执行失败", e);
            }
        }
    }

    /**
     * 提取文件基础名(不含扩展名)
     */
    private String extractBaseName(String fileName) {
        if (fileName == null) return "output";
        int idx = fileName.lastIndexOf('.');
        return idx > 0 ? fileName.substring(0, idx) : fileName;
    }

    /**
     * 删除文件
     */
    private void deleteFile(Path path) {
        if (path != null) {
            try {
                Files.deleteIfExists(path);
                log.trace("临时文件已删除: {}", path);
            } catch (Exception e) {
                log.warn("删除文件失败: {}", path, e);
            }
        }
    }

    /**
     * 关闭资源
     */
    private void closeResource(AutoCloseable resource) {
        if (resource != null) {
            try {
                resource.close();
            } catch (Exception e) {
                log.warn("关闭资源失败", e);
            }
        }
    }

    /**
     * 加载文件资源
     */
    public Resource loadFileAsResource(String fileName) throws Exception {
        Path filePath = uploadPath.resolve(fileName).normalize();
        Resource resource = new UrlResource(filePath.toUri());

        if (!resource.exists()) {
            throw new RuntimeException("文件未找到: " + fileName);
        }

        log.debug("文件资源已加载: {}", fileName);
        return resource;
    }

    // ==================== 内部数据类 ====================

    /**
     * 角度和置信度评分
     */
    private static class AngleScore {
        double angle;
        double score;

        AngleScore(double angle, double score) {
            this.angle = angle;
            this.score = score;
        }
    }

    /**
     * 文件准备结果
     */
    private static class FilePreparation {
        Path tempInputPath;
        Path outputPath;

        FilePreparation(Path tempInputPath, Path outputPath) {
            this.tempInputPath = tempInputPath;
            this.outputPath = outputPath;
        }
    }

    /**
     * 页面渲染结果
     */
    private static class PageRenderResult {
        final int pageIndex;
        final byte[] imageBytes;
        final boolean corrected;
        final PDRectangle originalSize;

        PageRenderResult(int idx, byte[] bytes, boolean c, PDRectangle size) {
            this.pageIndex = idx;
            this.imageBytes = bytes;
            this.corrected = c;
            this.originalSize = size;
        }
    }

    /**
     * 文本过滤参数
     */
    private static class FilterParams {
        int minHeight;
        int maxHeight;
        int minWidth;
        int maxWidth;
        double minArea;
        double maxArea;
        double minAspect;
        double maxAspect;
    }
}



























