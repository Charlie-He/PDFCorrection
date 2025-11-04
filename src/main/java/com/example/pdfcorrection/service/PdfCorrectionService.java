package com.example.pdfcorrection.service;

import com.example.pdfcorrection.model.CorrectedPageData;
import com.example.pdfcorrection.model.CorrectionResult;
import com.example.pdfcorrection.model.PageAngleResult;
import com.example.pdfcorrection.model.PageData;
import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
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
import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * PdfCorrectionService - 改进的倾斜检测与矫正*
 * 核心改进：
 * 1. Radon 变换使用相对标准差 (CV) 作为 score，避免量级失控
 * 2. 精细两阶段搜索：粗搜索 + 精搜索
 * 3. 优化融合策略，Hough 优先级更高
 * 4. 添加异常检测和回退机制
 */
@Service
@Slf4j
public class PdfCorrectionService {

    @Value("${file.upload-dir:uploads}")
    private String uploadDir;

    @Value("${pdf.correction.dpi:200}")  // 提高到 200 DPI
    private int renderDpi;

    @Value("${pdf.correction.min-angle}")
    private double minCorrectionAngle;

    @Value("${pdf.correction.compression.quality:0.92}")  // 提高压缩质量到 0.92
    private float compressionQuality;

    @Value("${pdf.correction.batch-size:10}")
    private int batchSize;

    @Value("${pdf.correction.use-png:false}")  // 新增：是否使用无损 PNG
    private boolean usePngFormat;

    @Value("${pdf.correction.preserve-content:false}")  // 新增：是否保留完整内容（会改变页面尺寸）
    private boolean preserveContent;

    // Tunables
    private static final int MAX_DETECTION_SIZE = 1000;

    // 移除信号量限制，让线程池控制并发
    // private static final Semaphore semaphore = new Semaphore(4);

    private final ExecutorService executorService;
    private static final AtomicInteger poolNumber = new AtomicInteger(1);

    public PdfCorrectionService() {
        int corePoolSize = Runtime.getRuntime().availableProcessors()-1;
        int maximumPoolSize = Runtime.getRuntime().availableProcessors();
        long keepAliveTime = 60L;
        // 限制队列大小，避免过多任务排队导致内存溢出
        BlockingQueue<Runnable> workingQueue = new ArrayBlockingQueue<>(16);
        ThreadFactory threadFactory = new ThreadFactory() {
            private final AtomicInteger threadNumber = new AtomicInteger(1);
            private final String namePrefix = "pdf-correction-pool-" + poolNumber.getAndIncrement() + "-thread-";

            @Override
            public Thread newThread(Runnable r) {
                Thread t = new Thread(r, namePrefix + threadNumber.getAndIncrement());
                t.setDaemon(false);
                t.setPriority(Thread.NORM_PRIORITY);
                return t;
            }
        };
        RejectedExecutionHandler handler = new ThreadPoolExecutor.CallerRunsPolicy();

        this.executorService = new ThreadPoolExecutor(
                corePoolSize,
                maximumPoolSize,
                keepAliveTime,
                TimeUnit.SECONDS,
                workingQueue,
                threadFactory,
                handler);
    }

    private Path uploadPath;

    @Autowired
    private ProgressService progressService;

    @PostConstruct
    public void init() {
        nu.pattern.OpenCV.loadLocally();
        uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
        try {
            Files.createDirectories(uploadPath);
        } catch (Exception e) {
            throw new RuntimeException("无法创建上传目录: " + uploadPath, e);
        }
        System.out.println("[PDF矫正服务] 初始化完成 - 改进的 Hough + Radon 模式");
    }

    @PreDestroy
    public void shutdownExecutorService() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
        System.out.println("[PDF矫正服务] 已关闭");
    }

    /**
     * 主流程：PDF 倾斜矫正（修复版本）
     */
    public CorrectionResult correctPdfSkewWithAngle(MultipartFile file) throws Exception {
        Path tempInputPath = null;
        Path outputPath;
        PDDocument document = null;
        PDDocument correctedDoc = null;

        long startTime = System.currentTimeMillis();

        try {
            String originalFileName = file.getOriginalFilename();
            String baseName = extractBaseName(originalFileName);
            String correctedFileName = baseName + "_corrected_" + UUID.randomUUID() + ".pdf";
            tempInputPath = uploadPath.resolve("temp_input_" + UUID.randomUUID() + ".pdf");
            outputPath = uploadPath.resolve(correctedFileName);
            file.transferTo(tempInputPath.toFile());

            document = PDDocument.load(tempInputPath.toFile());
            correctedDoc = new PDDocument();
            int numberOfPages = document.getNumberOfPages();
            progressService.sendProgress("开始处理 " + numberOfPages + " 页PDF...");

            List<Double> allAngles = new ArrayList<>();
            int totalBatches = (int) Math.ceil((double) numberOfPages / batchSize);

            for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
                int startPage = batchIndex * batchSize;
                int endPage = Math.min(startPage + batchSize, numberOfPages);

                long batchStartTime = System.currentTimeMillis();

                // 渲染阶段计时
                long renderStartTime = System.currentTimeMillis();
                List<PageData> batchPageData = renderPagesBatch(document, startPage, endPage);
                long renderTime = System.currentTimeMillis() - renderStartTime;
                progressService.sendProgress(String.format("批次 %d/%d (页面 %d-%d) 渲染完成，用时: %dms",
                        batchIndex + 1, totalBatches, startPage + 1, endPage, renderTime));

                // 角度检测阶段计时
                long detectionStartTime = System.currentTimeMillis();
                List<PageAngleResult> batchAngles = detectSkewAnglesBatch(batchPageData);
                long detectionTime = System.currentTimeMillis() - detectionStartTime;
                progressService.sendProgress(String.format("批次 %d/%d (页面 %d-%d) 角度检测完成，用时: %dms",
                        batchIndex + 1, totalBatches, startPage + 1, endPage, detectionTime));

                // 图像矫正阶段计时
                long correctionStartTime = System.currentTimeMillis();
                // 【关键修复】区分需要矫正和不需要矫正的页面
                List<CorrectedPageData> correctedPages = correctImagesBatchOptimized(
                        document, batchPageData, batchAngles, startPage);
                long correctionTime = System.currentTimeMillis() - correctionStartTime;
                progressService.sendProgress(String.format("批次 %d/%d (页面 %d-%d) 图像矫正完成，用时: %dms",
                        batchIndex + 1, totalBatches, startPage + 1, endPage, correctionTime));

                // 写入PDF阶段计时
                long writeStartTime = System.currentTimeMillis();
                writeCorrectedPagesToPdfOptimized(correctedDoc, document, correctedPages, startPage);
                long writeTime = System.currentTimeMillis() - writeStartTime;
                progressService.sendProgress(String.format("批次 %d/%d (页面 %d-%d) 写入PDF完成，用时: %dms",
                        batchIndex + 1, totalBatches, startPage + 1, endPage, writeTime));

                long batchTotalTime = System.currentTimeMillis() - batchStartTime;
                progressService.sendProgress(String.format("批次 %d/%d (页面 %d-%d) 总用时: %dms",
                        batchIndex + 1, totalBatches, startPage + 1, endPage, batchTotalTime));

                allAngles.addAll(batchAngles.stream().map(PageAngleResult::getAngle).toList());

                cleanupBatchResources(batchPageData, correctedPages);
                System.gc();
            }

            correctedDoc.save(outputPath.toFile());
            long totalTime = System.currentTimeMillis() - startTime;
            progressService.sendProgress("处理完成");
            progressService.sendProgress(String.format("总用时: %.2fs", totalTime / 1000.0));

            if (!allAngles.isEmpty()) {
                double avgAngle = allAngles.stream().mapToDouble(a -> a).average().orElse(0.0);
                progressService.sendAngleDetected(avgAngle);
            }

            return new CorrectionResult(outputPath.getFileName().toString(), allAngles, totalTime);

        } finally {
            closeResource(document);
            closeResource(correctedDoc);
            deleteFile(tempInputPath);
        }
    }

    /**
     * 多线程并行渲染PDF页面（优化版）
     */
    private List<PageData> renderPagesBatch(PDDocument document, int startPage, int endPage) {

        List<Future<?>> futures = new ArrayList<>();

        // 使用 ConcurrentHashMap 缓存渲染结果，保证页序正确
        ConcurrentHashMap<Integer, PageData> resultMap = new ConcurrentHashMap<>();

        for (int pageIndex = startPage; pageIndex < endPage; pageIndex++) {
            final int currentPage = pageIndex;
            futures.add(executorService.submit(() -> {
                try {
                    // 每个线程独立创建 PDFRenderer
                    PDFRenderer renderer = new PDFRenderer(document);
                    renderer.setSubsamplingAllowed(false);

                    // 动态DPI：避免A3或更大页面渲染过慢
                    PDPage page = document.getPage(currentPage);
                    float widthPt = page.getMediaBox().getWidth();
                    int adaptiveDpi = adaptDpi(widthPt);

                    BufferedImage image = renderer.renderImageWithDPI(
                            currentPage,
                            adaptiveDpi,
                            ImageType.RGB
                    );

                    resultMap.put(currentPage, new PageData(currentPage, image, page.getMediaBox()));
                } catch (Exception e) {
                    System.err.println("渲染第 " + (currentPage + 1) + " 页失败: " + e.getMessage());
                }
            }));
        }

        // 等待任务完成
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception ignored) {
            }
        }

        // 结果按页号排序
        return resultMap.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(Map.Entry::getValue)
                .toList();
    }

    /**
     * 根据页面宽度动态调整渲染DPI（防止A3页面爆内存）
     */
    private int adaptDpi(float widthPt) {
        if (widthPt > 800) return Math.min(renderDpi, 180); // A3或更大
        if (widthPt < 400) return Math.max(renderDpi, 220); // 小页略提DPI
        return renderDpi;
    }

    // ==================== 倾斜角检测（核心改进） ====================

    private List<PageAngleResult> detectSkewAnglesBatch(List<PageData> pageDataList) {
        List<CompletableFuture<PageAngleResult>> futures = pageDataList.stream()
                .map(pd -> CompletableFuture.supplyAsync(() -> {
                    return detectPageSkewAngle(pd);
                }, executorService))
                .toList();

        return futures.stream().map(CompletableFuture::join).collect(Collectors.toList());
    }

    /**
     * 单页检测：改进的融合策略
     */
    public PageAngleResult detectPageSkewAngle(PageData pageData) {
        Mat original = null;
        Mat gray = null;
        Mat processed = null;

        try {
            original = bufferedImageToMat(pageData.getImage());
            gray = preprocessForScannedDoc(original);

            double scale = calculateScaleFactor(gray);
            processed = new Mat();
            if (scale < 1.0) {
                Imgproc.resize(gray, processed, new Size(), scale, scale, Imgproc.INTER_AREA);
            } else {
                processed = gray.clone();
            }

            // === 分别计算 Hough 与 Radon ===
            AngleScore houghAS = detectByProbabilisticHoughWithScore(processed);
            AngleScore radonAS = detectByRadonOnTextMask(processed);

            // === 融合角度 ===
            double finalAngle = fuseTwoAngles(houghAS, radonAS);

            if (Math.abs(finalAngle) < minCorrectionAngle) {
                finalAngle = 0.0;
            }

            System.out.printf(
                    "第 %d 页 | Hough: %.2f° (score=%.0f) | Radon: %.2f° (score=%.3f) | 最终: %.2f°%n",
                    pageData.getPageIndex() + 1,
                    houghAS.angle, houghAS.score,
                    radonAS.angle, radonAS.score,
                    finalAngle
            );

            return new PageAngleResult(pageData.getPageIndex(), finalAngle);

        } catch (Exception e) {
            System.err.println("检测第 " + (pageData.getPageIndex() + 1) + " 页角度失败: " + e.getMessage());
            return new PageAngleResult(pageData.getPageIndex(), 0.0);
        } finally {
            releaseMat(original);
            releaseMat(gray);
            releaseMat(processed);
        }
    }

    // ==================== 优化后的 Hough（提升小角度精度） ====================

    private AngleScore detectByProbabilisticHoughWithScore(Mat gray) {
        Mat binary = new Mat();
        Mat horizontal = new Mat();
        Mat lines = new Mat();

        try {
            // 1. 二值化：用于形态学处理
            Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // 2. 增强文本行的连贯性（但避免过度平滑）
            // 仅使用水平膨胀：连接字符但避免模糊微小倾斜
            int kernelWidth = Math.max(20, gray.cols() / 60);
            Mat kernelH = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelWidth, 1)); // 1xWidth
            Imgproc.dilate(binary, horizontal, kernelH);
            kernelH.release();

            // 3. Canny 边缘检测：在增强后的图像上运行
            Imgproc.Canny(horizontal, horizontal, 50, 150, 3, false);

            // 4. 严格 HoughLinesP 参数
            int minLineLength = Math.max(gray.cols() / 6, 100); // 必须是长线段 (1/6 宽度)
            int maxLineGap = Math.max(gray.cols() / 15, 30);
            int threshold = 50; // 提高阈值，只检测高置信度线

            // theta 分辨率：提高精度，特别是针对小角度
            Imgproc.HoughLinesP(horizontal, lines, 1, Math.PI / 1080, threshold, minLineLength, maxLineGap); // 1°/180 -> 1°/540

            if (lines.rows() < 2) return new AngleScore(Double.NaN, 0.0);

            Map<Double, Double> bucketWeight = new HashMap<>();
            double bucketSize = 0.02;  // 桶尺寸：0.05 -> 0.02 (追求极限精度)

            for (int i = 0; i < lines.rows(); i++) {
                double[] l = lines.get(i, 0);
                double dx = l[2] - l[0];
                double dy = l[3] - l[1];
                double len = Math.hypot(dx, dy);

                double angle = Math.toDegrees(Math.atan2(dy, dx));
                angle = normalizeAngle(angle); // 确保在 [-45, 45]

                // 过滤：只接受非常接近水平的线段
                if (Math.abs(angle) > 5.0) continue; // 仅考虑 [-5.0, 5.0] 范围

                double bucket = Math.round(angle / bucketSize) * bucketSize;
                // 权重：使用长度的平方，让长线段拥有更大的话语权
                bucketWeight.merge(bucket, len * len, Double::sum);
            }

            if (bucketWeight.isEmpty()) return new AngleScore(Double.NaN, 0.0);

            // ... (后续加权平均逻辑不变，但由于 bucketSize 减小，精度已提高)

            // 找到得分最高的桶
            double bestBucket = bucketWeight.entrySet().stream()
                    .max(Map.Entry.comparingByValue()).get().getKey();

            // 加权平均（范围缩小）：只在最佳桶附近 ±0.1° 进行平均
            double weightedSum = 0.0, totalWeight = 0.0;
            double avgRange = 0.15; // 0.3 -> 0.15 (更聚焦)

            for (Map.Entry<Double, Double> entry : bucketWeight.entrySet()) {
                double bucket = entry.getKey();
                double weight = entry.getValue();

                if (Math.abs(bucket - bestBucket) <= avgRange) {
                    weightedSum += bucket * weight;
                    totalWeight += weight;
                }
            }

            double avgAngle = (totalWeight > 0) ? (weightedSum / totalWeight) : bestBucket;
            double score = bucketWeight.getOrDefault(bestBucket, 0.0);

            return new AngleScore(avgAngle, score);

        } catch (Exception e) {
            // ... (异常处理)
            return new AngleScore(Double.NaN, 0.0);
        } finally {
            releaseMat(binary);
            releaseMat(horizontal);
            releaseMat(lines);
        }
    }

    private AngleScore detectByRadonOnTextMask(Mat gray) {
        Mat textMask = null;
        Mat textOnly = null;
        Mat up = null;

        try {
            textMask = createTextMask(gray);
            if (textMask == null || Core.countNonZero(textMask) < (textMask.total() * 0.001)) {
                return detectByFullRadon(gray); // 如果掩码为空，回退到整体检测
            }

            textOnly = new Mat();
            gray.copyTo(textOnly, textMask);

            if (textOnly.empty() || Core.countNonZero(textOnly) == 0) {
                return detectByBlockwiseRadon(gray);  // 防止空区域检测
            }

            // 上采样：提升图像分辨率，捕捉微小角度变化
            double upScale = 2.0;  // 提升分辨率
            up = new Mat();
            Imgproc.resize(textOnly, up, new Size(), upScale, upScale, Imgproc.INTER_CUBIC);

            AngleScore blockAS = detectByBlockwiseRadon(up);

            // 如果块检测结果太弱，也用全图 Radon 进行补充确认
            if (blockAS.score < 0.5) {
                AngleScore fullAS = detectByFullRadon(gray);
                // 如果全图结果更好，则返回全图结果
                if (fullAS.score > blockAS.score) return fullAS;
            }
            return blockAS;

        } catch (Exception e) {
            System.err.println("Radon (TextMask) 检测异常: " + e.getMessage());
            return detectByBlockwiseRadon(gray);
        } finally {
            releaseMat(textMask);
            releaseMat(textOnly);
            releaseMat(up);
        }
    }

    // 新增函数：全图高精度 Radon
    private AngleScore detectByFullRadon(Mat gray) {
        Mat up = null;
        try {
            // 1. 强制上采样：即使原始图像是 TARGET_WIDTH，也将其放大以提高投影精度
            double upScale = 2.0;
            up = new Mat();
            Imgproc.resize(gray, up, new Size(), upScale, upScale, Imgproc.INTER_CUBIC);

            // 2. 严格控制 Radon 块检测的参数，但传入全图
            // detectRadonBlock 已经被优化到 0.01° 步长。
            return detectRadonBlock(up);
        } finally {
            releaseMat(up);
        }
    }




    /**
     * 核心：创建文本掩码 (方案 1.2)
     * 过滤掉所有非文本轮廓（如图形、线条），只保留“像”字符的轮廓。
     *
     * @param gray 原始灰度图
     * @return 一个 CV_8UC1 的掩码 Mat，其中“文本”区域为 255，其他为 0。
     */
    private Mat createTextMask(Mat gray) {
        Mat binary = new Mat();
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat textMask;

        try {
            // 1. 二值化 (为 findContours 准备)
            // 我们需要白色文本(255)，黑色背景(0)
            Imgproc.adaptiveThreshold(gray, binary, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 21, 5);


            // 2. 查找所有轮廓
            // RETR_LIST 模式更快，我们不需要层级关系
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            // 3. 准备掩码画布
            textMask = Mat.zeros(gray.size(), CvType.CV_8UC1);
            // 4. 启发式过滤：这是最关键的调优部分
            // ----------------------------------------------------
            // 这些参数取决于您渲染的 DPI (renderDpi)
            // 假设 DPI 约为 200-300
            //
            // 动态计算：我们使用图像高度来估计字体大小
            // 假设标准行高是页面高度的 1/100 到 1/20
            double scale = calculateScaleFactor(gray); // 复用您现有的缩放因子

            // 基于缩放动态调整参数
            int minCharHeight = (int) Math.max(8, 10 * scale);   // 最小字符高度 (过滤噪点)
            int maxCharHeight = (int) Math.max(100, 150 * scale); // 最大字符高度 (过滤大块图形)
            int minCharWidth = (int) Math.max(2, 3 * scale);
            int maxCharWidth = (int) Math.max(100, 150 * scale);
            double minArea = (int) Math.max(15, 20 * scale);        // 最小面积 (过滤噪点)
            double maxArea = (int) Math.max(2500, 3000 * scale);    // 最大面积 (过滤大块)
            double minAspect = 0.08; // 允许 'l' 或 'i'
            double maxAspect = 4.0;  // 允许 'm' 或 'w'
            // ----------------------------------------------------

            List<MatOfPoint> textContours = new ArrayList<>();

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);

                // --- 过滤规则 ---
                if (area < minArea || area > maxArea) {
                    continue; // 过滤掉太小（噪点）或太大（图片/表格块）
                }

                Rect rect = Imgproc.boundingRect(contour);

                if (rect.height < minCharHeight || rect.height > maxCharHeight) {
                    continue; // 过滤掉高度异常的
                }

                if (rect.width < minCharWidth || rect.width > maxCharWidth) {
                    continue; // 过滤掉宽度异常的
                }

                double aspectRatio = (double) rect.width / (double) rect.height;
                if (aspectRatio < minAspect || aspectRatio > maxAspect) {
                    continue; // 过滤掉长条形（如表格线）
                }

                // 通过所有测试，这可能是一个字符
                textContours.add(contour);
            }

            // 5. 将所有“字符”绘制到掩码上
            Imgproc.drawContours(textMask, textContours, -1, new Scalar(255), -1); // -1 = 填充

            // (可选) 为了连接断裂的字符，可以再做一次小的闭合
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            Imgproc.morphologyEx(textMask, textMask, Imgproc.MORPH_CLOSE, kernel);
            kernel.release();

            return textMask;

        } finally {
            // 释放中间资源
            releaseMat(binary);
            releaseMat(hierarchy);
            for (MatOfPoint contour : contours) {
                contour.release();
            }
        }
    }

    /**
     * 优化版 Blockwise Radon：
     * 1. detectRadonBlock: 增加内容检查，过滤掉空白或噪声块；缩小搜索范围以提速。
     * 2. detectByBlockwiseRadon: 不再使用中位数，而是使用加权投票直方图，
     * 让高置信度(score)的块在决定最终角度时有更大话语权。
     */
    private AngleScore detectByBlockwiseRadon(Mat gray) {
        int blockSize = Math.max(256, Math.min(gray.cols(), gray.rows()) / 8);
        int step = blockSize / 2;
        int rows = gray.rows(), cols = gray.cols();

        List<AngleScore> results = new ArrayList<>();

        // 1. 收集所有有效 block 的 AngleScore
        if (rows <= blockSize || cols <= blockSize) {
            results.add(detectRadonBlock(gray));
        } else {
            for (int y = 0; y + blockSize <= rows; y += step) {
                for (int x = 0; x + blockSize <= cols; x += step) {
                    Rect roi = new Rect(x, y, blockSize, blockSize);
                    Mat block = new Mat(gray, roi);
                    AngleScore as = detectRadonBlock(block);
                    block.release();
                    // 使用一个合理的最小置信度阈值
                    if (!Double.isNaN(as.angle) && as.score > 0.02) {
                        results.add(as);
                    }
                }
            }
        }

        if (results.isEmpty()) {
            return detectRadonBlock(gray); // Fallback
        }

        // 2. 加权投票直方图（代替中位数）
        Map<Double, Double> bucketWeight = new HashMap<>();
        double bucketSize = 0.2; // Radon 的桶可以细一点

        for (AngleScore as : results) {
            double bucket = Math.round(as.angle / bucketSize) * bucketSize;
            // 按 score 加权
            bucketWeight.merge(bucket, as.score, Double::sum);
        }

        if (bucketWeight.isEmpty()) return new AngleScore(Double.NaN, 0.0);

        // 3. 找到得分最高的桶
        double bestBucket = bucketWeight.entrySet().stream()
                .max(Map.Entry.comparingByValue()).get().getKey();

        // 4. 在最佳桶附近进行加权平均（同时平均分数）
        double weightedSum = 0.0, totalWeight = 0.0, scoreSum = 0.0, count = 0.0;
        for (AngleScore as : results) {
            double bucket = Math.round(as.angle / bucketSize) * bucketSize;
            if (Math.abs(bucket - bestBucket) <= 0.5) { // 附近 ±0.5°
                weightedSum += as.angle * as.score;
                totalWeight += as.score;
                scoreSum += as.score;
                count++;
            }
        }

        double avgAngle = (totalWeight > 0) ? (weightedSum / totalWeight) : bestBucket;
        double avgScore = (count > 0) ? (scoreSum / count) : 0.0;

        return new AngleScore(avgAngle, avgScore);
    }

    /**
     * Radon 块检测（优化版）
     */
// ---------- 3) detectRadonBlock：更细的角度步长以提高 <1° 分辨率 (优化版) ----------
    private AngleScore detectRadonBlock(Mat block) {
        Mat binary = new Mat();
        try {
            Imgproc.threshold(block, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            double nonZero = Core.countNonZero(binary);
            double ratio = nonZero / (double) (binary.rows() * binary.cols());
            if (ratio < 0.01 || ratio > 0.95) {
                return new AngleScore(Double.NaN, 0.0);
            }

            final double ANGLE_RANGE = 5.0; // 搜索范围保持不变

            // 改进：更小的步长来定位小角度
            final double COARSE_STEP = 0.05; // 0.1 -> 0.05
            final double FINE_STEP = 0.01;   // 0.02 -> 0.01 (更高精度)

            double bestAngle = 0.0, maxScore = -1.0;

            // 1. 粗略搜索
            for (double a = -ANGLE_RANGE; a <= ANGLE_RANGE; a += COARSE_STEP) {
                double score = calculateRadonCV(binary, a);
                if (score > maxScore) { maxScore = score; bestAngle = a; }
            }

            // 2. 精细搜索 (在最佳粗略角度附近 ±0.25° 范围)
            double fineStart = Math.max(-ANGLE_RANGE, bestAngle - 0.25); // 缩小精细搜索的范围
            double fineEnd = Math.min(ANGLE_RANGE, bestAngle + 0.25);

            for (double a = fineStart; a <= fineEnd; a += FINE_STEP) {
                double score = calculateRadonCV(binary, a);
                if (score > maxScore) { maxScore = score; bestAngle = a; }
            }

            // 3. 额外精细搜索 $0^\circ$ 附近（避免粗略搜索失误）
            // 如果粗略角度不在 $0^\circ$ 附近，我们仍然要精细化检查 $0^\circ$ 周围
            if (Math.abs(bestAngle) > 0.3) { // 仅当粗略角度较大时
                for (double a = -0.3; a <= 0.3; a += FINE_STEP) {
                    double score = calculateRadonCV(binary, a);
                    if (score > maxScore) { maxScore = score; bestAngle = a; }
                }
            }

            return new AngleScore(bestAngle, maxScore);

        } finally {
            releaseMat(binary);
        }
    }


    private double calculateRadonCV(Mat binary, double angle) {
        Mat rot = null, proj = null;
        MatOfDouble mean = null, std = null;
        try {
            rot = rotateImage(binary, angle);
            proj = new Mat();
            Core.reduce(rot, proj, 1, Core.REDUCE_SUM, CvType.CV_32F);

            mean = new MatOfDouble();
            std = new MatOfDouble();
            Core.meanStdDev(proj, mean, std);

            double meanVal = mean.get(0, 0)[0];
            double stdVal = std.get(0, 0)[0];
            if (meanVal < 1.0) return 0.0;

            return (stdVal * stdVal) / (meanVal + 1e-5); // CV²/mean 平滑化
        } finally {
            releaseMat(rot);
            releaseMat(proj);
            if (mean != null) mean.release();
            if (std != null) std.release();
        }
    }

    private double fuseTwoAngles(AngleScore h, AngleScore r) {
        boolean hNaN = (h == null) || Double.isNaN(h.angle);
        boolean rNaN = (r == null) || Double.isNaN(r.angle);
        if (hNaN && rNaN) return 0.0;
        if (hNaN) return r.angle;
        if (rNaN) return h.angle;

        double hScaled = Math.log1p(Math.max(0.0, h.score));
        double rScaled = Math.log1p(Math.max(0.0, r.score * 2.0));

        if (hScaled < 1e-6 && rScaled < 1e-6) return 0.0;

        // Softmax 权重计算 (保持不变)
        double maxv = Math.max(hScaled, rScaled);
        double hExp = Math.exp(hScaled - maxv);
        double rExp = Math.exp(rScaled - maxv);
        double wH = hExp / (hExp + rExp);
        double wR = rExp / (hExp + rExp);

        double delta = Math.abs(h.angle - r.angle);
        double hAbs = Math.abs(h.angle);
        double rAbs = Math.abs(r.angle);
        double meanAbs = (hAbs + rAbs) / 2.0;

        // --- 核心优化：放大非零角度的影响 ---

        // 1. 如果角度差异很小，但都非零（这是我们想要检测到的）
        if (delta < 0.2 && meanAbs > 0.1) {
            // 放大：对分数进行平方根处理，使高分结果权重更高
            wH = Math.sqrt(wH);
            wR = Math.sqrt(wR);
            // 如果两个结果都是负数（或正数），即方向一致，则进一步增强
            if (Math.signum(h.angle) == Math.signum(r.angle)) {
                wH *= 1.1;
                wR *= 1.1;
            }
        }
        // 2. 如果角度差异较大 (例如，Hough: 0.1, Radon: 0.5)
        else if (delta > 0.4) {
            // 如果其中一个结果接近 0.0，而另一个非零，我们应该更相信非零的结果
            if (hAbs < 0.2 && rAbs >= 0.2) {
                wR *= 1.5; // 增强非零的 Radon 结果
                wH *= 0.8;
            } else if (rAbs < 0.2 && hAbs >= 0.2) {
                wH *= 1.5; // 增强非零的 Hough 结果
                wR *= 0.8;
            } else {
                // 差异大且都不接近 0，相信分数更高的一方
                if (hScaled > rScaled) wH *= 1.1;
                else wR *= 1.1;
            }
        }

        // 最终归一化
        double sum = wH + wR;
        if (sum <= 0) return (h.angle + r.angle) / 2.0;
        wH /= sum;
        wR /= sum;

        return h.angle * wH + r.angle * wR;
    }

    // ==================== 优化的批量矫正方法 ====================

    /**
     * 优化版本：对于不需要矫正的页面，直接复制原始页面，避免重渲染
     */
    private List<CorrectedPageData> correctImagesBatchOptimized(
            PDDocument sourceDoc,
            List<PageData> pageDataList,
            List<PageAngleResult> angleResults,
            int startPageIndex) {

        List<CorrectedPageData> corrected = new ArrayList<>();
        for (int i = 0; i < pageDataList.size(); i++) {
            PageData pd = pageDataList.get(i);
            double angle = (i < angleResults.size()) ? angleResults.get(i).getAngle() : 0.0;

            // 标记是否需要矫正
            boolean needsCorrection = Math.abs(angle) > minCorrectionAngle;

            if (needsCorrection) {
                // 需要矫正：渲染并旋转图像
                BufferedImage corr = correctImageSkew(pd.getImage(), angle);
                corrected.add(new CorrectedPageData(pd.getPageIndex(), corr, pd.getOriginalSize(), true));
            } else {
                // 不需要矫正：标记为直接复制原始页面
                corrected.add(new CorrectedPageData(pd.getPageIndex(), null, pd.getOriginalSize(), false));
            }
        }
        return corrected;
    }

    private void writeCorrectedPagesToPdfOptimized(
            PDDocument targetDoc,
            PDDocument sourceDoc,
            List<CorrectedPageData> correctedPages,
            int startPageIndex) throws Exception {

        // Step 1️⃣ 并行编码阶段
        List<CompletableFuture<PageRenderResult>> futures = correctedPages.stream()
                .map(pageData -> CompletableFuture.supplyAsync(() -> {
                    try {
                        if (!pageData.needsCorrection()) {
                            // 无需矫正的页面，不处理图像
                            return new PageRenderResult(pageData.getPageIndex(), null, false, pageData.getOriginalSize());
                        }

                        BufferedImage image = pageData.getImage();
                        PDRectangle originalSize = pageData.getOriginalSize();

                        // --- 图像转字节数组 ---
                        ByteArrayOutputStream baos = new ByteArrayOutputStream(1024 * 1024);
                        if (usePngFormat) {
                            ImageIO.write(image, "PNG", baos);
                        } else {
                            ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
                            ImageWriteParam params = writer.getDefaultWriteParam();
                            params.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                            params.setCompressionQuality(Math.min(Math.max(compressionQuality, 0.5f), 0.95f));
                            writer.setOutput(ImageIO.createImageOutputStream(baos));
                            writer.write(null, new IIOImage(image, null, null), params);
                            writer.dispose();
                        }

                        byte[] imgBytes = baos.toByteArray();
                        return new PageRenderResult(pageData.getPageIndex(), imgBytes, true, originalSize);

                    } catch (Exception e) {
                        log.error("⚠ 第 {} 页图像编码失败: {}", pageData.getPageIndex() + 1, e.getMessage());
                        return new PageRenderResult(pageData.getPageIndex(), null, false, pageData.getOriginalSize());
                    }
                }, executorService))
                .toList();

        // 等待全部完成
        List<PageRenderResult> results = futures.stream()
                .map(CompletableFuture::join)
                .sorted(Comparator.comparingInt(r -> r.pageIndex)) // ✅ 按页序排序
                .toList();

        // Step 2️⃣ 顺序写入阶段
        for (PageRenderResult result : results) {
            int pageIndex = result.pageIndex;
            if (!result.corrected || result.imageBytes == null) {
                // 无需矫正的页面
                PDPage originalPage = sourceDoc.getPage(pageIndex);
                targetDoc.importPage(originalPage);
                log.debug("✓ 第 {} 页无需矫正，已复制", pageIndex + 1);
                continue;
            }

            PDRectangle size = result.originalSize;
            float pdfWidth = size.getWidth();
            float pdfHeight = size.getHeight();

            PDPage pdPage = new PDPage(new PDRectangle(pdfWidth, pdfHeight));
            targetDoc.addPage(pdPage);

            PDImageXObject pdImage = PDImageXObject.createFromByteArray(
                    targetDoc, result.imageBytes, usePngFormat ? "page.png" : "page.jpg");

            try (PDPageContentStream contentStream = new PDPageContentStream(
                    targetDoc, pdPage, PDPageContentStream.AppendMode.OVERWRITE, true, true)) {
                contentStream.drawImage(pdImage, 0, 0, pdfWidth, pdfHeight);
            }

//            log.debug("第 {} 页已写入 (size={}×{})", pageIndex + 1, pdfWidth, pdfHeight);
        }

        log.info("所有 {} 页写入完成", results.size());
    }

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



    // ==================== 预处理、矫正与输出 ====================

    private Mat preprocessForScannedDoc(Mat original) {
        Mat gray = new Mat();
        if (original.channels() >= 3) {
            Imgproc.cvtColor(original, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = original.clone();
        }

        Imgproc.medianBlur(gray, gray, 3);
        return gray;
    }

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
                // 模式1：扩大边界保留完整内容（页面尺寸会变化）
                double radians = Math.toRadians(angle);
                double sin = Math.abs(Math.sin(radians));
                double cos = Math.abs(Math.cos(radians));

                int newWidth = (int) Math.ceil(mat.width() * cos + mat.height() * sin);
                int newHeight = (int) Math.ceil(mat.width() * sin + mat.height() * cos);

                // 调整平移量，使图像居中
                double[] tx = rotMat.get(0, 2);
                double[] ty = rotMat.get(1, 2);
                tx[0] += (newWidth - mat.cols()) / 2.0;
                ty[0] += (newHeight - mat.rows()) / 2.0;
                rotMat.put(0, 2, tx);
                rotMat.put(1, 2, ty);

                Imgproc.warpAffine(mat, rotated, rotMat, new Size(newWidth, newHeight),
                        Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));
            } else {
                // 模式2：保持原始尺寸（推荐，页面大小一致）
                Imgproc.warpAffine(mat, rotated, rotMat, mat.size(),
                        Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));
            }

            rotMat.release();
            return matToBufferedImage(rotated);
        } finally {
            releaseMat(mat);
            releaseMat(rotated);
        }
    }

    // ==================== 辅助函数 ====================

    private double calculateScaleFactor(Mat mat) {
        final double MIN_SCALE = 0.5; // 确保缩放因子不低于 0.5

        int longEdge = Math.max(mat.cols(), mat.rows());

        if (longEdge <= MAX_DETECTION_SIZE) {
            return 1.0;
        }

        double scale = (double) MAX_DETECTION_SIZE / (double) longEdge;

        // 限制最小缩放，防止丢失过多细节
        return Math.max(scale, MIN_SCALE);
    }

    private double normalizeAngle(double angle) {
        while (angle > 45) angle -= 90;
        while (angle < -45) angle += 90;
        return angle;
    }

    private Mat bufferedImageToMat(BufferedImage image) {
        if (image == null) return new Mat();
        BufferedImage converted = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = converted.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();
        byte[] pixels = ((DataBufferByte) converted.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(converted.getHeight(), converted.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, pixels);
        return mat;
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) return null;
        int type = mat.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        byte[] data = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, data);
        byte[] target = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, target, 0, data.length);
        return image;
    }

    /**
     * 对二值/文档图像友好的旋转函数（无裁切、保持边缘锐利）
     */
    private Mat rotateImage(Mat src, double angle) {
        int w = src.cols();
        int h = src.rows();
        Point center = new Point(w / 2.0, h / 2.0);

        // 计算旋转矩阵
        Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        // 计算旋转后图像需要的完整包围框大小（避免裁切）
        double absCos = Math.abs(rotMat.get(0, 0)[0]);
        double absSin = Math.abs(rotMat.get(0, 1)[0]);
        int newW = (int) Math.round(h * absSin + w * absCos);
        int newH = (int) Math.round(h * absCos + w * absSin);

        // 调整平移量（将旋转后中心对齐）
        rotMat.put(0, 2, rotMat.get(0, 2)[0] + (newW / 2.0 - center.x));
        rotMat.put(1, 2, rotMat.get(1, 2)[0] + (newH / 2.0 - center.y));

        // 输出图像
        Mat dst = new Mat();
        int interp = Imgproc.INTER_NEAREST; // 防止文字模糊
        Scalar bgColor = src.channels() == 1 ? new Scalar(255) : new Scalar(255, 255, 255);

        Imgproc.warpAffine(src, dst, rotMat, new Size(newW, newH), interp, Core.BORDER_CONSTANT, bgColor);

        rotMat.release();
        return dst;
    }


    private void releaseMat(Mat mat) {
        if (mat != null && !mat.empty()) mat.release();
    }

    private void cleanupBatchResources(List<PageData> pageDataList, List<CorrectedPageData> correctedPages) {
        for (PageData pd : pageDataList) {
            if (pd.getImage() != null) pd.getImage().flush();
        }
        for (CorrectedPageData cp : correctedPages) {
            if (cp.getImage() != null) cp.getImage().flush();
        }
    }

    private String extractBaseName(String fileName) {
        if (fileName == null) return "output";
        int idx = fileName.lastIndexOf('.');
        return idx > 0 ? fileName.substring(0, idx) : fileName;
    }

    private void deleteFile(Path p) {
        if (p != null) {
            try { Files.deleteIfExists(p); } catch (Exception ignored) {}
        }
    }

    private void closeResource(AutoCloseable r) {
        if (r != null) {
            try { r.close(); } catch (Exception ignored) {}
        }
    }

    public Resource loadFileAsResource(String fileName) throws Exception {
        Path filePath = uploadPath.resolve(fileName).normalize();
        Resource resource = new UrlResource(filePath.toUri());
        if (!resource.exists()) throw new RuntimeException("文件未找到: " + fileName);
        return resource;
    }


    private static class AngleScore {
        double angle;
        double score;
        AngleScore(double angle, double score) {
            this.angle = angle;
            this.score = score;
        }
    }
}