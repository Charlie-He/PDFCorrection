package com.example.pdfcorrection.service;

import com.example.pdfcorrection.model.PageData;
import com.example.pdfcorrection.model.PageAngleResult;
import com.example.pdfcorrection.model.CorrectedPageData;
import com.example.pdfcorrection.model.CorrectionResult;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * 增强版PDF倾斜矫正服务
 * 采用多策略自适应算法，针对不同类型的页面选择最优检测方法
 *
 * 主要优化：
 * 1. 梯度方向直方图作为主算法（快速准确）
 * 2. 概率霍夫变换+聚类分析（处理复杂布局）
 * 3. 自适应预处理（应对低质量扫描）
 * 4. 多方法投票机制（提高鲁棒性）
 * 5. 文本密度分析（区分文本页和图像页）
 *
 * @author Enhanced Version
 */
@Service
public class PdfCorrectionService {

    @Value("${file.upload-dir:uploads}")
    private String uploadDir;

    @Value("${pdf.correction.dpi:150}")
    private int renderDpi;

    @Value("${pdf.correction.min-angle:0.15}")
    private double minCorrectionAngle;

    @Value("${pdf.correction.compression.quality:0.8}")
    private float compressionQuality;

    // 检测时的最大图像长边
    private static final int MAX_DETECTION_SIZE = 1200;

    // 角度精度
    private static final double ANGLE_PRECISION = 0.1;

    // 文本密度阈值（用于判断页面类型）
    private static final double TEXT_DENSITY_THRESHOLD = 0.05;

    // 线程池
    private final ExecutorService executorService = Executors.newFixedThreadPool(
            Math.max(2, Runtime.getRuntime().availableProcessors() - 1));

    private Path uploadPath;

    @Autowired
    private ProgressService progressService;

    @PostConstruct
    public void init() {
        // 加载OpenCV
        nu.pattern.OpenCV.loadLocally();

        uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
        try {
            Files.createDirectories(uploadPath);
        } catch (Exception e) {
            throw new RuntimeException("无法创建上传目录: " + uploadPath, e);
        }

        System.out.println("[PDF矫正服务] 初始化完成 - 多策略自适应算法");
    }

    /**
     * 主流程：PDF倾斜矫正
     */
    public CorrectionResult correctPdfSkewWithAngle(MultipartFile file) throws Exception {
        Path tempInputPath = null;
        Path outputPath = null;
        PDDocument document = null;
        PDDocument correctedDoc = null;

        long startTime = System.currentTimeMillis();

        try {
            // 1. 准备文件路径
            String originalFileName = file.getOriginalFilename();
            String baseName = extractBaseName(originalFileName);
            String correctedFileName = baseName + "_corrected_" + UUID.randomUUID() + ".pdf";
            tempInputPath = uploadPath.resolve("temp_input_" + UUID.randomUUID() + ".pdf");
            outputPath = uploadPath.resolve(correctedFileName);
            file.transferTo(tempInputPath.toFile());

            // 2. 加载PDF文档
            document = PDDocument.load(tempInputPath.toFile());
            correctedDoc = new PDDocument();
            int numberOfPages = document.getNumberOfPages();
            System.out.println("开始处理 " + numberOfPages + " 页PDF...");
            progressService.sendProgress("开始处理 " + numberOfPages + " 页PDF...");

            // 3. 并行渲染所有页面
            long renderStart = System.currentTimeMillis();
            List<PageData> pageDataList = renderPagesParallel(document, numberOfPages);
            long renderTime = System.currentTimeMillis() - renderStart;
            double renderTimeSeconds = renderTime / 1000.0;
            System.out.println("渲染完成，用时: " + String.format("%.2f", renderTimeSeconds) + "s");
            progressService.sendProgress("渲染完成，用时: " + String.format("%.2f", renderTimeSeconds) + "s");

            // 4. 并行检测倾斜角度（自适应多策略）
            long detectStart = System.currentTimeMillis();
            List<PageAngleResult> angleResults = detectSkewAnglesParallel(pageDataList);
            long detectTime = System.currentTimeMillis() - detectStart;
            double detectTimeSeconds = detectTime / 1000.0;
            System.out.println("角度检测完成，用时: " + String.format("%.2f", detectTimeSeconds) + "s");
            progressService.sendProgress("角度检测完成，用时: " + String.format("%.2f", detectTimeSeconds) + "s");

            // 发送检测到的角度信息
            if (!angleResults.isEmpty()) {
                double avgAngle = angleResults.stream()
                        .mapToDouble(PageAngleResult::getAngle)
                        .average()
                        .orElse(0.0);
                progressService.sendAngleDetected(avgAngle);
            }

            // 5. 并行矫正图像
            long correctStart = System.currentTimeMillis();
            List<CorrectedPageData> correctedPages = correctImagesParallel(pageDataList, angleResults);
            long correctTime = System.currentTimeMillis() - correctStart;
            double correctTimeSeconds = correctTime / 1000.0;
            System.out.println("图像矫正完成，用时: " + String.format("%.2f", correctTimeSeconds) + "s");
            progressService.sendProgress("图像矫正完成，用时: " + String.format("%.2f", correctTimeSeconds) + "s");

            // 6. 批量写入新PDF文档（优化性能）
            long writeStart = System.currentTimeMillis();
            writeCorrectedPagesToPdf(correctedDoc, correctedPages);
            correctedDoc.save(outputPath.toFile());
            long writeTime = System.currentTimeMillis() - writeStart;
            double writeTimeSeconds = writeTime / 1000.0;
            System.out.println("写入PDF完成，用时: " + String.format("%.2f", writeTimeSeconds) + "s");
            progressService.sendProgress("写入PDF完成，用时: " + String.format("%.2f", writeTimeSeconds) + "s");

            long totalTime = System.currentTimeMillis() - startTime;
            double totalTimeSeconds = totalTime / 1000.0;
            System.out.println("总用时: " + String.format("%.2f", totalTimeSeconds) + "s");
            progressService.sendProgress("总用时: " + String.format("%.2f", totalTimeSeconds) + "s");

            // 7. 收集角度信息
            List<Double> angles = angleResults.stream()
                    .map(PageAngleResult::getAngle)
                    .collect(Collectors.toList());

            return new CorrectionResult(correctedFileName, angles);

        } finally {
            closeResource(document);
            closeResource(correctedDoc);
            deleteFile(tempInputPath);
        }
    }

    // ==================== 阶段1: 并行渲染 ====================

    private List<PageData> renderPagesParallel(PDDocument document, int numberOfPages) {
        List<CompletableFuture<PageData>> futures = new ArrayList<>();

        for (int i = 0; i < numberOfPages; i++) {
            final int pageIndex = i;
            CompletableFuture<PageData> future = CompletableFuture.supplyAsync(() -> {
                try {
                    PDFRenderer localRenderer = new PDFRenderer(document);
                    BufferedImage image = localRenderer.renderImageWithDPI(pageIndex, renderDpi);
                    PDPage page = document.getPage(pageIndex);
                    return new PageData(pageIndex, image, page.getMediaBox());
                } catch (Exception e) {
                    throw new RuntimeException("渲染第 " + (pageIndex + 1) + " 页失败", e);
                }
            }, executorService);
            futures.add(future);
        }

        return futures.stream()
                .map(CompletableFuture::join)
                .sorted(Comparator.comparingInt(PageData::getPageIndex))
                .collect(Collectors.toList());
    }

    // ==================== 阶段2: 并行检测（增强版） ====================

    private List<PageAngleResult> detectSkewAnglesParallel(List<PageData> pageDataList) {
        List<CompletableFuture<PageAngleResult>> futures = pageDataList.stream()
                .map(pd -> CompletableFuture.supplyAsync(() -> detectPageSkewAngle(pd), executorService))
                .collect(Collectors.toList());

        return futures.stream()
                .map(CompletableFuture::join)
                .sorted(Comparator.comparingInt(PageAngleResult::getPageIndex))
                .collect(Collectors.toList());
    }

    /**
     * 增强版单页倾斜角度检测
     * 根据页面特征自适应选择最优算法组合
     */
    private PageAngleResult detectPageSkewAngle(PageData pageData) {
        Mat original = null;
        Mat resized = null;
        Mat gray = null;
        Mat enhanced = null;

        try {
            // 1. 图像预处理
            original = bufferedImageToMat(pageData.getImage());
            double scale = calculateScaleFactor(original);
            resized = resizeImage(original, scale);

            // 转换为灰度图
            gray = new Mat();
            if (resized.channels() >= 3) {
                Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = resized.clone();
            }

            // 2. 自适应图像增强（针对低质量扫描）
            enhanced = enhanceImageQuality(gray);

            // 3. 分析页面特征
            PageFeatures features = analyzePageFeatures(enhanced);

            // 4. 多策略检测并投票
            List<AngleCandidate> candidates = new ArrayList<>();

            // 策略1: 梯度方向直方图（主算法，适合大部分情况）
            double gradientAngle = detectSkewByGradientHistogram(enhanced, features);
            if (!Double.isNaN(gradientAngle)) {
                candidates.add(new AngleCandidate(gradientAngle, 3.0, "Gradient"));
            }

            // 策略2: 概率霍夫变换+聚类（适合复杂布局和多方向文本）
            if (features.hasComplexLayout || features.textDensity < TEXT_DENSITY_THRESHOLD) {
                double houghAngle = detectSkewByProbabilisticHough(enhanced, features);
                if (!Double.isNaN(houghAngle)) {
                    candidates.add(new AngleCandidate(houghAngle, 2.5, "PHough"));
                }
            }

            // 策略3: 投影法（适合文本密度高的页面）
            if (features.textDensity >= TEXT_DENSITY_THRESHOLD) {
                double projAngle = detectSkewByProjection(enhanced, features);
                if (!Double.isNaN(projAngle)) {
                    candidates.add(new AngleCandidate(projAngle, 2.0, "Projection"));
                }
            }

            // 策略4: 傅里叶变换（备选方法，适合周期性文本）
            if (candidates.isEmpty() || features.textDensity > 0.1) {
                double fourierAngle = detectSkewByFourierTransform(enhanced);
                if (!Double.isNaN(fourierAngle)) {
                    candidates.add(new AngleCandidate(fourierAngle, 1.5, "Fourier"));
                }
            }

            // 5. 加权投票选择最终角度
            double finalAngle = 0.0;
            if (!candidates.isEmpty()) {
                finalAngle = weightedVoting(candidates);

                // 精细调整
                if (Math.abs(finalAngle) > minCorrectionAngle) {
                    finalAngle = refineAngle(enhanced, finalAngle);
                }
            }

            // 6. 小角度过滤
            if (Math.abs(finalAngle) < minCorrectionAngle) {
                finalAngle = 0.0;
            }

            // 7. 日志输出
            String candidateInfo = candidates.stream()
                    .map(c -> String.format("%s:%.2f°(w%.1f)", c.method, c.angle, c.weight))
                    .collect(Collectors.joining(", "));

            System.out.println(String.format(
                    "第 %d 页 | 角度: %.3f° | 文本密度: %.3f | 候选: [%s]",
                    pageData.getPageIndex() + 1, finalAngle, features.textDensity, candidateInfo
            ));

            return new PageAngleResult(pageData.getPageIndex(), finalAngle);

        } catch (Exception e) {
            System.err.println("检测角度异常: " + e.getMessage());
            e.printStackTrace();
            return new PageAngleResult(pageData.getPageIndex(), 0.0);
        } finally {
            releaseMat(original);
            releaseMat(resized);
            releaseMat(gray);
            releaseMat(enhanced);
        }
    }

    /**
     * 自适应图像质量增强
     * 针对低质量扫描进行预处理
     */
    private Mat enhanceImageQuality(Mat gray) {
        Mat enhanced = new Mat();

        // 1. 自适应直方图均衡化（CLAHE）- 增强对比度
        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        clahe.apply(gray, enhanced);

        // 2. 双边滤波 - 降噪同时保留边缘
        Mat filtered = new Mat();
        Imgproc.bilateralFilter(enhanced, filtered, 5, 50, 50);
        enhanced.release();

        return filtered;
    }

    /**
     * 分析页面特征
     */
    private PageFeatures analyzePageFeatures(Mat gray) {
        PageFeatures features = new PageFeatures();

        // 1. 计算文本密度
        Mat binary = new Mat();
        Imgproc.adaptiveThreshold(gray, binary, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 15, 10);

        int nonZero = Core.countNonZero(binary);
        features.textDensity = (double) nonZero / (binary.rows() * binary.cols());

        // 2. 检测是否有复杂布局（多列、表格等）
        Mat edges = new Mat();
        Imgproc.Canny(binary, edges, 50, 150);

        // 检测水平和垂直线
        Mat linesH = new Mat();
        Mat linesV = new Mat();
        Imgproc.HoughLinesP(edges, linesH, 1, Math.PI / 180, 50, gray.cols() / 4, 20);
        Imgproc.HoughLinesP(edges, linesV, 1, Math.PI / 180, 50, gray.rows() / 4, 20);

        // 如果检测到多条长线，可能是复杂布局
        features.hasComplexLayout = (linesH.rows() > 5 || linesV.rows() > 5);

        // 3. 估算文本行数
        int[] rowProj = new int[binary.rows()];
        for (int r = 0; r < binary.rows(); r++) {
            for (int c = 0; c < binary.cols(); c++) {
                if (binary.get(r, c)[0] > 0) {
                    rowProj[r]++;
                }
            }
        }

        // 计算峰值数量（文本行数估计）
        int peakCount = 0;
        int threshold = (int)(binary.cols() * 0.1);
        for (int i = 1; i < rowProj.length - 1; i++) {
            if (rowProj[i] > threshold &&
                    rowProj[i] > rowProj[i-1] &&
                    rowProj[i] > rowProj[i+1]) {
                peakCount++;
            }
        }
        features.estimatedTextLines = peakCount;

        binary.release();
        edges.release();
        linesH.release();
        linesV.release();

        return features;
    }

    /**
     * 策略1: 增强版梯度方向直方图
     */
    private double detectSkewByGradientHistogram(Mat gray, PageFeatures features) {
        Mat gradX = null;
        Mat gradY = null;

        try {
            // 计算梯度
            gradX = new Mat();
            gradY = new Mat();
            Imgproc.Sobel(gray, gradX, CvType.CV_32F, 1, 0, 3);
            Imgproc.Sobel(gray, gradY, CvType.CV_32F, 0, 1, 3);

            // 构建直方图（-45° 到 +45°）
            int numBins = 180;
            double[] histogram = new double[numBins];

            // 自适应采样步长
            int step = Math.max(1, Math.min(gradX.rows(), gradX.cols()) / 400);

            // 自适应梯度阈值（根据文本密度调整）
            double magThreshold = features.textDensity < TEXT_DENSITY_THRESHOLD ? 20 : 30;

            for (int y = 0; y < gradX.rows(); y += step) {
                for (int x = 0; x < gradX.cols(); x += step) {
                    float gx = (float) gradX.get(y, x)[0];
                    float gy = (float) gradY.get(y, x)[0];

                    double magnitude = Math.sqrt(gx * gx + gy * gy);

                    if (magnitude > magThreshold) {
                        double angle = Math.toDegrees(Math.atan2(gy, gx)) - 90;
                        angle = normalizeAngle(angle);

                        int bin = (int)((angle + 45) * 2);
                        if (bin >= 0 && bin < numBins) {
                            histogram[bin] += magnitude;
                        }
                    }
                }
            }

            // 平滑直方图
            histogram = smoothHistogram(histogram, 5);

            // 找峰值
            int maxBin = 0;
            double maxValue = histogram[0];
            for (int i = 1; i < numBins; i++) {
                if (histogram[i] > maxValue) {
                    maxValue = histogram[i];
                    maxBin = i;
                }
            }

            // 检查峰值显著性
            double avgValue = Arrays.stream(histogram).average().orElse(0);
            if (maxValue < avgValue * 1.3) {
                return Double.NaN;
            }

            // 子像素精度（抛物线拟合）
            double angle = (maxBin / 2.0) - 45.0;
            if (maxBin > 0 && maxBin < numBins - 1) {
                double prev = histogram[maxBin - 1];
                double curr = histogram[maxBin];
                double next = histogram[maxBin + 1];
                double offset = 0.5 * (prev - next) / (prev - 2 * curr + next);
                angle = ((maxBin + offset) / 2.0) - 45.0;
            }

            return angle;

        } finally {
            releaseMat(gradX);
            releaseMat(gradY);
        }
    }

    /**
     * 策略2: 概率霍夫变换+聚类分析
     * 适用于复杂布局和包含图像的页面
     */
    private double detectSkewByProbabilisticHough(Mat gray, PageFeatures features) {
        Mat binary = null;
        Mat edges = null;
        Mat lines = null;

        try {
            // 1. 二值化
            binary = new Mat();
            Imgproc.adaptiveThreshold(gray, binary, 255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY_INV, 15, 10);

            // 2. 形态学操作 - 连接文本行
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 2));
            Mat closed = new Mat();
            Imgproc.morphologyEx(binary, closed, Imgproc.MORPH_CLOSE, kernel);
            kernel.release();

            // 3. 边缘检测
            edges = new Mat();
            Imgproc.Canny(closed, edges, 50, 150);
            closed.release();

            // 4. 概率霍夫变换
            lines = new Mat();
            int minLineLength = Math.max(50, gray.cols() / 8);
            Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, 60, minLineLength, 15);

            if (lines.rows() < 3) {
                return Double.NaN;
            }

            // 5. 提取线段角度并过滤
            List<Double> angles = new ArrayList<>();
            for (int i = 0; i < lines.rows(); i++) {
                double[] l = lines.get(i, 0);
                double dx = l[2] - l[0];
                double dy = l[3] - l[1];
                double len = Math.hypot(dx, dy);

                // 过滤短线段
                if (len < minLineLength * 0.8) continue;

                double angle = Math.toDegrees(Math.atan2(dy, dx));
                angle = normalizeAngle(angle);

                // 只保留±45°以内
                if (Math.abs(angle) <= 45) {
                    angles.add(angle);
                }
            }

            if (angles.isEmpty()) {
                return Double.NaN;
            }

            // 6. 聚类分析找主要方向
            return clusterAndFindMainAngle(angles);

        } finally {
            releaseMat(binary);
            releaseMat(edges);
            releaseMat(lines);
        }
    }

    /**
     * 聚类分析找主要角度
     * 使用改进的均值漂移算法
     */
    private double clusterAndFindMainAngle(List<Double> angles) {
        if (angles.isEmpty()) return Double.NaN;

        Collections.sort(angles);

        // 简单聚类：查找密度最大的区域
        double bandwidth = 2.0; // 2度的带宽
        Map<Double, Integer> clusters = new HashMap<>();

        for (double angle : angles) {
            double center = Math.round(angle / bandwidth) * bandwidth;
            clusters.put(center, clusters.getOrDefault(center, 0) + 1);
        }

        // 找最大聚类
        double maxCenter = 0.0;
        int maxCount = 0;
        for (Map.Entry<Double, Integer> entry : clusters.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                maxCenter = entry.getKey();
            }
        }

        // 计算该聚类的精确平均值
        double finalMaxCenter = maxCenter;
        List<Double> clusterAngles = angles.stream()
                .filter(a -> Math.abs(a - finalMaxCenter) <= bandwidth)
                .collect(Collectors.toList());

        return clusterAngles.stream().mapToDouble(a -> a).average().orElse(0.0);
    }

    /**
     * 策略3: 投影法（适用于文本密度高的页面）
     */
    private double detectSkewByProjection(Mat gray, PageFeatures features) {
        Mat binary = null;
        Mat small = null;

        try {
            // 二值化
            binary = new Mat();
            Imgproc.adaptiveThreshold(gray, binary, 255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY_INV, 15, 10);

            // 缩小加速
            small = new Mat();
            Imgproc.resize(binary, small, new Size(
                    Math.max(350, binary.cols() / 2.0),
                    Math.max(350, binary.rows() / 2.0)
            ));

            // 粗略搜索
            double bestAngle = 0.0;
            double maxVar = Double.NEGATIVE_INFINITY;

            for (double angle = -15; angle <= 15; angle += 0.8) {
                Mat rotated = rotateImage(small, angle);
                double variance = calculateProjectionVariance(rotated);
                rotated.release();

                if (variance > maxVar) {
                    maxVar = variance;
                    bestAngle = angle;
                }
            }

            return bestAngle;

        } finally {
            releaseMat(binary);
            releaseMat(small);
        }
    }

    /**
     * 策略4: 傅里叶变换检测
     * 基于频谱分析的方法，适合周期性文本
     */
    private double detectSkewByFourierTransform(Mat gray) {
        Mat resized = null;
        Mat floatImg = null;
        Mat dft = null;

        try {
            // 1. 缩小图像
            resized = new Mat();
            Imgproc.resize(gray, resized, new Size(256, 256));

            // 2. 转换为浮点型
            floatImg = new Mat();
            resized.convertTo(floatImg, CvType.CV_32F);

            // 3. DFT
            dft = new Mat();
            Core.dft(floatImg, dft, Core.DFT_COMPLEX_OUTPUT, 0);

            // 4. 计算幅度谱
            List<Mat> planes = new ArrayList<>();
            Core.split(dft, planes);
            Core.magnitude(planes.get(0), planes.get(1), planes.get(0));
            Mat magnitude = planes.get(0);

            // 5. 对数变换
            Core.add(magnitude, Scalar.all(1), magnitude);
            Core.log(magnitude, magnitude);

            // 6. 裁剪和重排（移动零频到中心）
            magnitude = magnitude.submat(
                    0, magnitude.rows() & -2,
                    0, magnitude.cols() & -2
            );

            int cx = magnitude.cols() / 2;
            int cy = magnitude.rows() / 2;

            Mat q0 = new Mat(magnitude, new Rect(0, 0, cx, cy));
            Mat q1 = new Mat(magnitude, new Rect(cx, 0, cx, cy));
            Mat q2 = new Mat(magnitude, new Rect(0, cy, cx, cy));
            Mat q3 = new Mat(magnitude, new Rect(cx, cy, cx, cy));

            Mat tmp = new Mat();
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);

            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);

            // 7. 径向投影找主方向
            double bestAngle = 0.0;
            double maxIntensity = Double.NEGATIVE_INFINITY;

            for (double angle = -45; angle <= 45; angle += 1.0) {
                double intensity = calculateRadialProjection(magnitude, angle);
                if (intensity > maxIntensity) {
                    maxIntensity = intensity;
                    bestAngle = angle;
                }
            }

            planes.get(0).release();
            planes.get(1).release();
            tmp.release();

            return bestAngle;

        } catch (Exception e) {
            return Double.NaN;
        } finally {
            releaseMat(resized);
            releaseMat(floatImg);
            releaseMat(dft);
        }
    }

    /**
     * 计算径向投影强度
     */
    private double calculateRadialProjection(Mat magnitude, double angle) {
        int cx = magnitude.cols() / 2;
        int cy = magnitude.rows() / 2;
        double rad = Math.toRadians(angle);
        double cos = Math.cos(rad);
        double sin = Math.sin(rad);

        double sum = 0;
        int count = 0;
        int maxRadius = Math.min(cx, cy) - 5;

        for (int r = 10; r < maxRadius; r++) {
            int x = cx + (int)(r * cos);
            int y = cy + (int)(r * sin);

            if (x >= 0 && x < magnitude.cols() && y >= 0 && y < magnitude.rows()) {
                sum += magnitude.get(y, x)[0];
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    /**
     * 加权投票选择最终角度
     * 根据各方法的权重和角度一致性进行投票
     */
    private double weightedVoting(List<AngleCandidate> candidates) {
        if (candidates.isEmpty()) return 0.0;
        if (candidates.size() == 1) return candidates.get(0).angle;

        // 1. 如果所有候选角度都很接近（标准差小），使用加权平均
        double[] angles = candidates.stream().mapToDouble(c -> c.angle).toArray();
        double mean = Arrays.stream(angles).average().orElse(0.0);
        double variance = Arrays.stream(angles)
                .map(a -> (a - mean) * (a - mean))
                .average()
                .orElse(0.0);
        double stdDev = Math.sqrt(variance);

        // 如果角度一致性高（标准差<2°），使用加权平均
        if (stdDev < 2.0) {
            double weightedSum = 0;
            double totalWeight = 0;
            for (AngleCandidate c : candidates) {
                weightedSum += c.angle * c.weight;
                totalWeight += c.weight;
            }
            return weightedSum / totalWeight;
        }

        // 2. 如果角度分散，选择权重最高的
        return candidates.stream()
                .max(Comparator.comparingDouble(c -> c.weight))
                .map(c -> c.angle)
                .orElse(0.0);
    }

    /**
     * 精细调整角度
     */
    private double refineAngle(Mat gray, double coarseAngle) {
        Mat binary = null;
        Mat small = null;

        try {
            // 二值化
            binary = new Mat();
            Imgproc.adaptiveThreshold(gray, binary, 255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY_INV, 15, 10);

            // 适度缩小
            small = new Mat();
            Imgproc.resize(binary, small, new Size(
                    Math.max(450, binary.cols() / 1.5),
                    Math.max(450, binary.rows() / 1.5)
            ));

            // 精细搜索：±1.5°，步长0.1°
            double bestAngle = coarseAngle;
            double maxVar = Double.NEGATIVE_INFINITY;
            double start = coarseAngle - 1.5;
            double end = coarseAngle + 1.5;

            for (double angle = start; angle <= end; angle += ANGLE_PRECISION) {
                Mat rotated = rotateImage(small, angle);
                double variance = calculateProjectionVariance(rotated);
                rotated.release();

                if (variance > maxVar) {
                    maxVar = variance;
                    bestAngle = angle;
                }
            }

            return bestAngle;

        } finally {
            releaseMat(binary);
            releaseMat(small);
        }
    }

    /**
     * 计算投影方差
     */
    private double calculateProjectionVariance(Mat mat) {
        int cols = mat.cols();
        int rows = mat.rows();

        if (cols <= 0 || rows <= 0) {
            return Double.NEGATIVE_INFINITY;
        }

        // 列采样步长
        int step = Math.max(1, cols / 100);
        List<Double> projection = new ArrayList<>();

        // 计算每列的投影和
        for (int c = 0; c < cols; c += step) {
            double sum = 0;
            for (int r = 0; r < rows; r++) {
                double[] pixel = mat.get(r, c);
                sum += (pixel != null && pixel.length > 0) ? pixel[0] : 0;
            }
            projection.add(sum);
        }

        // 计算方差
        double mean = projection.stream().mapToDouble(d -> d).average().orElse(0.0);
        double variance = projection.stream()
                .mapToDouble(d -> (d - mean) * (d - mean))
                .average()
                .orElse(0.0);

        return variance;
    }

    /**
     * 平滑直方图
     */
    private double[] smoothHistogram(double[] histogram, int windowSize) {
        int len = histogram.length;
        double[] smoothed = new double[len];
        int half = windowSize / 2;

        for (int i = 0; i < len; i++) {
            double sum = 0;
            int count = 0;
            for (int j = -half; j <= half; j++) {
                int idx = i + j;
                if (idx >= 0 && idx < len) {
                    sum += histogram[idx];
                    count++;
                }
            }
            smoothed[i] = sum / count;
        }

        return smoothed;
    }

    // ==================== 阶段3: 并行矫正 ====================

    private List<CorrectedPageData> correctImagesParallel(
            List<PageData> pageDataList,
            List<PageAngleResult> angleResults) {

        List<CompletableFuture<CorrectedPageData>> futures = new ArrayList<>();

        for (int i = 0; i < pageDataList.size(); i++) {
            PageData pd = pageDataList.get(i);
            double angle = angleResults.get(i).getAngle();

            CompletableFuture<CorrectedPageData> future = CompletableFuture.supplyAsync(() -> {
                BufferedImage corrected = correctImageSkew(pd.getImage(), angle);
                pd.getImage().flush();
                return new CorrectedPageData(pd.getPageIndex(), corrected, pd.getOriginalSize());
            }, executorService);

            futures.add(future);
        }

        return futures.stream()
                .map(CompletableFuture::join)
                .sorted(Comparator.comparingInt(CorrectedPageData::getPageIndex))
                .collect(Collectors.toList());
    }

    /**
     * 矫正图像倾斜
     */
    private BufferedImage correctImageSkew(BufferedImage image, double angle) {
        if (image == null || Math.abs(angle) < minCorrectionAngle) {
            return image;
        }

        Mat mat = null;
        Mat rotated = null;

        try {
            mat = bufferedImageToMat(image);

            // 计算旋转后的新尺寸
            double rad = Math.toRadians(Math.abs(angle));
            int newW = (int) Math.ceil(mat.cols() * Math.cos(rad) + mat.rows() * Math.sin(rad));
            int newH = (int) Math.ceil(mat.cols() * Math.sin(rad) + mat.rows() * Math.cos(rad));

            // 获取旋转矩阵
            Point center = new Point(mat.cols() / 2.0, mat.rows() / 2.0);
            Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0);

            // 调整平移量
            double[] tx = rotationMatrix.get(0, 2);
            double[] ty = rotationMatrix.get(1, 2);
            tx[0] += (newW - mat.cols()) / 2.0;
            ty[0] += (newH - mat.rows()) / 2.0;
            rotationMatrix.put(0, 2, tx);
            rotationMatrix.put(1, 2, ty);

            // 执行仿射变换
            rotated = new Mat();
            Imgproc.warpAffine(mat, rotated, rotationMatrix, new Size(newW, newH),
                    Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));

            rotationMatrix.release();

            return matToBufferedImage(rotated);

        } catch (Exception e) {
            System.err.println("矫正图像失败: " + e.getMessage());
            return image;
        } finally {
            releaseMat(mat);
            releaseMat(rotated);
        }
    }

    // ==================== 工具方法 ====================

    /**
     * 旋转图像
     */
    private Mat rotateImage(Mat src, double angle) {
        Point center = new Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Mat dst = new Mat();
        Imgproc.warpAffine(src, dst, rotationMatrix, src.size(),
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
        rotationMatrix.release();
        return dst;
    }

    /**
     * 角度归一化
     */
    private double normalizeAngle(double angle) {
        while (angle > 45) angle -= 90;
        while (angle < -45) angle += 90;
        return angle;
    }

    /**
     * 计算缩放因子
     */
    private double calculateScaleFactor(Mat mat) {
        int longEdge = Math.max(mat.cols(), mat.rows());
        return (longEdge > MAX_DETECTION_SIZE) ?
                (double) MAX_DETECTION_SIZE / (double) longEdge : 1.0;
    }

    /**
     * 缩放图像
     */
    private Mat resizeImage(Mat src, double scale) {
        if (Math.abs(scale - 1.0) < 1e-6) {
            return src.clone();
        }

        Mat out = new Mat();
        Imgproc.resize(src, out, new Size(src.cols() * scale, src.rows() * scale),
                0, 0, Imgproc.INTER_AREA);
        return out;
    }

    /**
     * BufferedImage 转 Mat
     */
    private Mat bufferedImageToMat(BufferedImage image) {
        if (image == null) {
            return new Mat();
        }

        BufferedImage converted = new BufferedImage(
                image.getWidth(),
                image.getHeight(),
                BufferedImage.TYPE_3BYTE_BGR
        );

        Graphics2D g = converted.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();

        byte[] pixels = ((DataBufferByte) converted.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(converted.getHeight(), converted.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, pixels);

        return mat;
    }

    /**
     * Mat 转 BufferedImage
     */
    private BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        int type = mat.channels() == 1 ?
                BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);

        byte[] data = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, data);

        byte[] target = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, target, 0, data.length);

        return image;
    }

    /**
     * 写入矫正后的页面到PDF - 最终优化版本
     */
    private void writeCorrectedPageToPdf(PDDocument doc, CorrectedPageData pageData)
            throws IOException {

        BufferedImage image = pageData.getImage();
        PDRectangle originalSize = pageData.getOriginalSize();

        PDPage pdPage = new PDPage(new PDRectangle(
                originalSize.getWidth(),
                originalSize.getHeight()
        ));
        doc.addPage(pdPage);

        // 使用JPEG格式以提高性能，牺牲一些图像质量
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
        ImageWriteParam params = writer.getDefaultWriteParam();
        params.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        params.setCompressionQuality(compressionQuality); // 使用可配置的压缩质量

        writer.setOutput(ImageIO.createImageOutputStream(baos));
        writer.write(null, new IIOImage(image, null, null), params);
        writer.dispose();

        PDImageXObject pdImage = PDImageXObject.createFromByteArray(doc, baos.toByteArray(), "page.jpg");

        try (PDPageContentStream contentStream = new PDPageContentStream(doc, pdPage,
                PDPageContentStream.AppendMode.APPEND, true, true)) {
            contentStream.drawImage(pdImage, 0, 0,
                    originalSize.getWidth(), originalSize.getHeight());
        }

        image.flush();
    }

    /**
     * 批量写入矫正后的页面到PDF - 性能优化版本
     */
    private void writeCorrectedPagesToPdf(PDDocument doc, List<CorrectedPageData> correctedPages)
            throws IOException {

        for (CorrectedPageData pageData : correctedPages) {
            BufferedImage image = pageData.getImage();
            PDRectangle originalSize = pageData.getOriginalSize();

            PDPage pdPage = new PDPage(new PDRectangle(
                    originalSize.getWidth(),
                    originalSize.getHeight()
            ));
            doc.addPage(pdPage);

            // 使用JPEG格式以提高性能，牺牲一些图像质量
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
            ImageWriteParam params = writer.getDefaultWriteParam();
            params.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            params.setCompressionQuality(compressionQuality); // 使用可配置的压缩质量

            writer.setOutput(ImageIO.createImageOutputStream(baos));
            writer.write(null, new IIOImage(image, null, null), params);
            writer.dispose();

            PDImageXObject pdImage = PDImageXObject.createFromByteArray(doc, baos.toByteArray(), "page.jpg");

            try (PDPageContentStream contentStream = new PDPageContentStream(doc, pdPage,
                    PDPageContentStream.AppendMode.APPEND, true, true)) {
                contentStream.drawImage(pdImage, 0, 0,
                        originalSize.getWidth(), originalSize.getHeight());
            }

            image.flush();
        }
    }

    private void releaseMat(Mat m) {
        if (m != null && !m.empty()) {
            m.release();
        }
    }

    private String extractBaseName(String fileName) {
        if (fileName == null) {
            return "output";
        }
        int idx = fileName.lastIndexOf('.');
        return idx > 0 ? fileName.substring(0, idx) : fileName;
    }

    private void deleteFile(Path p) {
        if (p != null) {
            try {
                Files.deleteIfExists(p);
            } catch (Exception ignored) {}
        }
    }

    private void closeResource(AutoCloseable r) {
        if (r != null) {
            try {
                r.close();
            } catch (Exception ignored) {}
        }
    }

    public Resource loadFileAsResource(String fileName) throws Exception {
        Path filePath = uploadPath.resolve(fileName).normalize();
        Resource resource = new UrlResource(filePath.toUri());

        if (!resource.exists()) {
            throw new RuntimeException("文件未找到: " + fileName);
        }

        return resource;
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

    // ==================== 内部类 ====================

    /**
     * 页面特征
     */
    private static class PageFeatures {
        double textDensity = 0.0;           // 文本密度
        boolean hasComplexLayout = false;   // 是否有复杂布局
        int estimatedTextLines = 0;         // 估算文本行数
    }

    /**
     * 角度候选
     */
    private static class AngleCandidate {
        double angle;      // 角度值
        double weight;     // 权重
        String method;     // 检测方法

        AngleCandidate(double angle, double weight, String method) {
            this.angle = angle;
            this.weight = weight;
            this.method = method;
        }
    }
}