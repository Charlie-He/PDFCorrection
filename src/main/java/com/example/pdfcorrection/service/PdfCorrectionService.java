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
import org.opencv.imgproc.Imgproc;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * PDF倾斜矫正服务 - 优化版
 *
 * 采用混合算法策略：
 * 1. 快速投影分析 - 初步估计倾斜角度
 * 2. Hough线检测 - 验证和精确角度
 * 3. 投影方差优化 - 最终角度微调
 *
 * 特性：
 * - 并行处理多页PDF
 * - 智能缓存和资源管理
 * - 自适应阈值调整
 * - 保持原有API接口
 */
@Service
public class PdfCorrectionService {

    @Value("${file.upload-dir:uploads}")
    private String uploadDir;

    @Value("${pdf.correction.dpi:150}")
    private int renderDpi;

    @Value("${pdf.correction.min-angle:0.1}")
    private double minCorrectionAngle;

    private Path uploadPath;

    // 线程池：使用可用处理器数-1，保留一个核心给系统
    private final ExecutorService executorService = Executors.newFixedThreadPool(
            Math.max(2, Runtime.getRuntime().availableProcessors() - 1));

    @PostConstruct
    public void init() {
        // 加载OpenCV本地库
        nu.pattern.OpenCV.loadLocally();

        uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
        try {
            Files.createDirectories(uploadPath);
            cleanupTempFiles();
        } catch (Exception e) {
            throw new RuntimeException("无法创建上传目录", e);
        }
    }

    /**
     * 定时清理临时文件（每小时执行一次）
     */
    @Scheduled(fixedRate = 3600000)
    public void scheduledCleanup() {
        cleanupTempFiles();
    }

    /**
     * 清理超过1小时的临时文件
     */
    private void cleanupTempFiles() {
        try {
            Files.list(uploadPath)
                    .filter(path -> {
                        String name = path.getFileName().toString();
                        return name.startsWith("temp_");
                    })
                    .filter(path -> {
                        try {
                            long age = System.currentTimeMillis() - Files.getLastModifiedTime(path).toMillis();
                            return age > 3600000; // 1小时
                        } catch (IOException e) {
                            return false;
                        }
                    })
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (Exception ignored) {}
                    });
        } catch (Exception e) {
            System.err.println("清理临时文件失败: " + e.getMessage());
        }
    }

    /**
     * 主流程：PDF倾斜矫正
     * 保持原有API接口不变
     */
    public CorrectionResult correctPdfSkewWithAngle(MultipartFile file) throws Exception {
        Path tempInputPath = null;
        Path outputPath = null;
        PDDocument document = null;
        PDDocument correctedDoc = null;

        long startTime = System.currentTimeMillis();

        try {
            // 生成输出文件名
            String originalFileName = file.getOriginalFilename();
            String baseName = extractBaseName(originalFileName);
            String correctedFileName = baseName + "_corrected_" + UUID.randomUUID() + ".pdf";

            // 保存上传的文件
            tempInputPath = uploadPath.resolve("temp_input_" + UUID.randomUUID() + ".pdf");
            outputPath = uploadPath.resolve(correctedFileName);
            file.transferTo(tempInputPath.toFile());

            // 加载PDF文档
            document = PDDocument.load(tempInputPath.toFile());
            correctedDoc = new PDDocument();
            int numberOfPages = document.getNumberOfPages();

            System.out.println("开始处理 " + numberOfPages + " 页PDF...");

            // 步骤1：渲染所有页面为图像
            PDFRenderer renderer = new PDFRenderer(document);
            List<PageData> pageDataList = new ArrayList<>();

            for (int i = 0; i < numberOfPages; i++) {
                PDPage page = document.getPage(i);
                BufferedImage image = renderer.renderImageWithDPI(i, renderDpi);
                pageDataList.add(new PageData(i, image, page.getMediaBox()));
            }

            System.out.println("渲染完成，用时: " + (System.currentTimeMillis() - startTime) + "ms");

            // 步骤2：并行检测每页的倾斜角度
            long detectStart = System.currentTimeMillis();
            List<CompletableFuture<PageAngleResult>> angleFutures = pageDataList.stream()
                    .map(pageData -> CompletableFuture.supplyAsync(() ->
                            detectPageSkewAngle(pageData), executorService))
                    .collect(Collectors.toList());

            List<PageAngleResult> angleResults = angleFutures.stream()
                    .map(CompletableFuture::join)
                    .sorted(Comparator.comparingInt(PageAngleResult::getPageIndex))
                    .collect(Collectors.toList());

            System.out.println("角度检测完成，用时: " + (System.currentTimeMillis() - detectStart) + "ms");

            // 步骤3：并行矫正每页图像
            long correctStart = System.currentTimeMillis();
            List<CompletableFuture<CorrectedPageData>> correctionFutures = new ArrayList<>();

            for (int i = 0; i < pageDataList.size(); i++) {
                PageData pageData = pageDataList.get(i);
                double angle = angleResults.get(i).getAngle();

                CompletableFuture<CorrectedPageData> future = CompletableFuture.supplyAsync(() -> {
                    BufferedImage corrected = correctImageSkew(pageData.getImage(), angle);
                    pageData.getImage().flush(); // 释放原图像
                    return new CorrectedPageData(pageData.getPageIndex(), corrected, pageData.getOriginalSize());
                }, executorService);

                correctionFutures.add(future);
            }

            List<CorrectedPageData> correctedPages = correctionFutures.stream()
                    .map(CompletableFuture::join)
                    .sorted(Comparator.comparingInt(CorrectedPageData::getPageIndex))
                    .collect(Collectors.toList());

            System.out.println("图像矫正完成，用时: " + (System.currentTimeMillis() - correctStart) + "ms");

            // 步骤4：写入新PDF文档
            long writeStart = System.currentTimeMillis();
            for (CorrectedPageData correctedPage : correctedPages) {
                writeCorrectedPageToPdf(correctedDoc, correctedPage);
            }
            correctedDoc.save(outputPath.toFile());

            System.out.println("写入PDF完成，用时: " + (System.currentTimeMillis() - writeStart) + "ms");
            System.out.println("总用时: " + (System.currentTimeMillis() - startTime) + "ms");

            // 提取角度列表
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

    /**
     * 检测单页的倾斜角度
     *
     * 算法流程：
     * 1. 预处理：灰度化、自适应二值化
     * 2. 快速投影分析：粗略估计角度范围
     * 3. Hough线检测：识别主要文本线
     * 4. 投影方差优化：精确角度
     */
    private PageAngleResult detectPageSkewAngle(PageData pageData) {
        Mat mat = null;
        Mat gray = null;
        Mat binary = null;

        try {
            mat = bufferedImageToMat(pageData.getImage());

            // 降采样以提高速度（保持宽度在1200像素以内）
            double scale = Math.min(1.0, 1200.0 / mat.cols());
            Mat resized = new Mat();
            if (scale < 1.0) {
                Imgproc.resize(mat, resized, new Size(mat.cols() * scale, mat.rows() * scale));
            } else {
                resized = mat.clone();
            }

            // 图像预处理
            gray = new Mat();
            if (resized.channels() == 3 || resized.channels() == 4) {
                Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = resized.clone();
            }

            // 高斯模糊去噪
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(gray, blurred, new Size(3, 3), 0);

            // 自适应二值化
            binary = new Mat();
            Imgproc.adaptiveThreshold(blurred, binary, 255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 15, 10);

            // 形态学操作：连接文本行
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(30, 2));
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);
            kernel.release();

            // 步骤1：快速投影分析（-15°到+15°，步长1°）
            double coarseAngle = fastProjectionAnalysis(binary);

            // 步骤2：Hough线检测验证
            double houghAngle = houghLineDetection(binary, resized.cols());

            // 步骤3：选择最佳角度并进行精细优化
            double initialAngle = selectBestAngle(coarseAngle, houghAngle);
            double refinedAngle = refineAngleByProjection(binary, initialAngle);

            // 清理资源
            resized.release();
            blurred.release();

            // 如果角度太小，返回0（不需要矫正）
            if (Math.abs(refinedAngle) < minCorrectionAngle) {
                refinedAngle = 0.0;
            }

            System.out.println(String.format("第%d页检测角度: %.3f° (粗略: %.3f°, Hough: %.3f°)",
                    pageData.getPageIndex() + 1, refinedAngle, coarseAngle, houghAngle));

            return new PageAngleResult(pageData.getPageIndex(), refinedAngle);

        } catch (Exception e) {
            System.err.println("检测角度失败: " + e.getMessage());
            return new PageAngleResult(pageData.getPageIndex(), 0.0);
        } finally {
            releaseMat(mat);
            releaseMat(gray);
            releaseMat(binary);
        }
    }

    /**
     * 快速投影分析
     * 通过计算不同角度下的列投影方差，找到方差最大的角度
     * 方差越大表示文本行对齐越好
     */
    private double fastProjectionAnalysis(Mat binary) {
        // 进一步降采样以提高速度
        Mat small = new Mat();
        Imgproc.resize(binary, small, new Size(
                Math.max(100, binary.cols() / 4),
                Math.max(100, binary.rows() / 4)));

        double bestAngle = 0.0;
        double maxVariance = Double.NEGATIVE_INFINITY;

        // 在-15°到+15°范围内搜索，步长1°
        for (int angle = -15; angle <= 15; angle++) {
            Mat rotated = rotateImage(small, angle);
            double variance = calculateProjectionVariance(rotated);

            if (variance > maxVariance) {
                maxVariance = variance;
                bestAngle = angle;
            }

            rotated.release();
        }

        small.release();
        return bestAngle;
    }

    /**
     * Hough线检测
     * 检测图像中的主要直线，统计其角度
     */
    private double houghLineDetection(Mat binary, int imageWidth) {
        Mat edges = new Mat();
        Imgproc.Canny(binary, edges, 50, 150, 3);

        Mat lines = new Mat();
        // 动态调整参数：根据图像宽度调整最小线长
        int minLineLength = Math.max(imageWidth / 10, 50);
        Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, 50, minLineLength, 20);

        List<Double> angles = new ArrayList<>();

        if (lines.rows() > 0) {
            // 最多检查200条线
            int maxLines = Math.min(lines.rows(), 200);

            for (int i = 0; i < maxLines; i++) {
                double[] line = lines.get(i, 0);
                double dx = line[2] - line[0];
                double dy = line[3] - line[1];

                // 过滤太短的线
                double length = Math.hypot(dx, dy);
                if (length < imageWidth * 0.1) {
                    continue;
                }

                // 计算角度（相对于水平线）
                double angle = Math.toDegrees(Math.atan2(dy, dx));

                // 标准化角度到[-45, 45]范围
                angle = normalizeAngle(angle);

                // 只保留接近水平的线（±30°以内）
                if (Math.abs(angle) <= 30) {
                    angles.add(angle);
                }
            }
        }

        edges.release();
        lines.release();

        // 如果没有检测到线，返回0
        if (angles.isEmpty()) {
            return 0.0;
        }

        // 使用中位数并过滤离群值
        return calculateRobustMedian(angles);
    }

    /**
     * 选择最佳角度
     * 综合投影分析和Hough检测的结果
     */
    private double selectBestAngle(double projectionAngle, double houghAngle) {
        // 如果两个方法的结果相近（差异小于2°），使用Hough结果（通常更准确）
        if (Math.abs(projectionAngle - houghAngle) < 2.0) {
            return houghAngle;
        }

        // 如果Hough检测到明显的倾斜，优先使用
        if (Math.abs(houghAngle) > 0.5) {
            return houghAngle;
        }

        // 否则使用投影分析结果
        return projectionAngle;
    }

    /**
     * 精细角度优化
     * 在初始角度附近进行小范围精细搜索（步长0.1°）
     */
    private double refineAngleByProjection(Mat binary, double initialAngle) {
        Mat small = new Mat();
        Imgproc.resize(binary, small, new Size(
                Math.max(200, binary.cols() / 2),
                Math.max(200, binary.rows() / 2)));

        double bestAngle = initialAngle;
        double maxVariance = Double.NEGATIVE_INFINITY;

        // 在初始角度±2°范围内搜索，步长0.1°
        double start = initialAngle - 2.0;
        double end = initialAngle + 2.0;

        for (double angle = start; angle <= end; angle += 0.1) {
            Mat rotated = rotateImage(small, angle);
            double variance = calculateProjectionVariance(rotated);

            if (variance > maxVariance) {
                maxVariance = variance;
                bestAngle = angle;
            }

            rotated.release();
        }

        small.release();
        return bestAngle;
    }

    /**
     * 计算列投影方差
     * 方差越大表示文本行越对齐
     */
    private double calculateProjectionVariance(Mat mat) {
        int cols = mat.cols();
        int rows = mat.rows();

        if (cols <= 0 || rows <= 0) {
            return Double.NEGATIVE_INFINITY;
        }

        // 采样列以提高速度（最多采样100列）
        int step = Math.max(1, cols / 100);
        List<Double> projections = new ArrayList<>();

        for (int c = 0; c < cols; c += step) {
            double colSum = 0;
            for (int r = 0; r < rows; r++) {
                double[] pixel = mat.get(r, c);
                colSum += (pixel != null && pixel.length > 0) ? pixel[0] : 0;
            }
            projections.add(colSum);
        }

        // 计算方差
        double mean = projections.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = projections.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);

        return variance;
    }

    /**
     * 旋转图像
     */
    private Mat rotateImage(Mat src, double angle) {
        org.opencv.core.Point center = new org.opencv.core.Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Mat dst = new Mat();
        Imgproc.warpAffine(src, dst, rotMat, src.size(),
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
        rotMat.release();
        return dst;
    }

    /**
     * 标准化角度到[-45, 45]范围
     */
    private double normalizeAngle(double angle) {
        while (angle > 90) angle -= 180;
        while (angle < -90) angle += 180;
        if (angle > 45) angle -= 90;
        if (angle < -45) angle += 90;
        return angle;
    }

    /**
     * 计算鲁棒中位数（过滤离群值）
     */
    private double calculateRobustMedian(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }

        if (values.size() < 4) {
            Collections.sort(values);
            return values.get(values.size() / 2);
        }

        // 排序
        Collections.sort(values);

        // 计算四分位数
        int n = values.size();
        int q1Index = n / 4;
        int q3Index = n * 3 / 4;
        double q1 = values.get(q1Index);
        double q3 = values.get(q3Index);
        double iqr = q3 - q1;

        // IQR方法过滤离群值
        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;

        List<Double> filtered = values.stream()
                .filter(v -> v >= lowerBound && v <= upperBound)
                .collect(Collectors.toList());

        if (filtered.isEmpty()) {
            return values.get(n / 2);
        }

        return filtered.get(filtered.size() / 2);
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

            // 计算旋转后的新尺寸（避免裁剪）
            double radians = Math.toRadians(Math.abs(angle));
            int newWidth = (int) Math.ceil(mat.cols() * Math.cos(radians) + mat.rows() * Math.sin(radians));
            int newHeight = (int) Math.ceil(mat.cols() * Math.sin(radians) + mat.rows() * Math.cos(radians));

            // 创建旋转矩阵
            org.opencv.core.Point center = new org.opencv.core.Point(mat.cols() / 2.0, mat.rows() / 2.0);
            Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);

            // 调整平移量以容纳完整图像
            double[] tx = rotMat.get(0, 2);
            double[] ty = rotMat.get(1, 2);
            tx[0] += (newWidth - mat.cols()) / 2.0;
            ty[0] += (newHeight - mat.rows()) / 2.0;
            rotMat.put(0, 2, tx);
            rotMat.put(1, 2, ty);

            // 执行旋转（使用白色背景）
            rotated = new Mat();
            Imgproc.warpAffine(mat, rotated, rotMat, new Size(newWidth, newHeight),
                    Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));

            rotMat.release();

            BufferedImage result = matToBufferedImage(rotated);
            return result;

        } catch (Exception e) {
            System.err.println("矫正图像失败: " + e.getMessage());
            return image;
        } finally {
            releaseMat(mat);
            releaseMat(rotated);
        }
    }

    /**
     * 将矫正后的页面写入PDF
     */
    private void writeCorrectedPageToPdf(PDDocument doc, CorrectedPageData pageData) throws IOException {
        BufferedImage image = pageData.getImage();
        PDRectangle originalSize = pageData.getOriginalSize();

        // 创建新页面（使用原始页面尺寸）
        PDPage pdPage = new PDPage(new PDRectangle(originalSize.getWidth(), originalSize.getHeight()));
        doc.addPage(pdPage);

        // 临时保存图像
        Path tempImagePath = uploadPath.resolve("temp_page_" + UUID.randomUUID() + ".png");

        try {
            ImageIO.write(image, "PNG", tempImagePath.toFile());
            PDImageXObject pdImage = PDImageXObject.createFromFile(tempImagePath.toString(), doc);

            // 将图像绘制到页面
            try (PDPageContentStream contentStream = new PDPageContentStream(doc, pdPage)) {
                contentStream.drawImage(pdImage, 0, 0, originalSize.getWidth(), originalSize.getHeight());
            }
        } finally {
            deleteFile(tempImagePath);
            image.flush();
        }
    }

    /**
     * BufferedImage转Mat（BGR格式）
     */
    private Mat bufferedImageToMat(BufferedImage image) {
        if (image == null) {
            return new Mat();
        }

        BufferedImage converted = new BufferedImage(
                image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = converted.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();

        byte[] pixels = ((DataBufferByte) converted.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(converted.getHeight(), converted.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, pixels);

        return mat;
    }

    /**
     * Mat转BufferedImage
     */
    private BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        int type = mat.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);

        byte[] data = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, data);

        byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, targetPixels, 0, data.length);

        return image;
    }

    /**
     * 从文件名提取基本名称（不含扩展名）
     */
    private String extractBaseName(String fileName) {
        if (fileName == null) {
            return "output";
        }
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
    }

    /**
     * 安全释放Mat资源
     */
    private void releaseMat(Mat mat) {
        if (mat != null && !mat.empty()) {
            mat.release();
        }
    }

    /**
     * 关闭AutoCloseable资源
     */
    private void closeResource(AutoCloseable resource) {
        if (resource != null) {
            try {
                resource.close();
            } catch (Exception ignored) {}
        }
    }

    /**
     * 删除文件
     */
    private void deleteFile(Path path) {
        if (path != null) {
            try {
                Files.deleteIfExists(path);
            } catch (Exception ignored) {}
        }
    }

    // ==================== 公共API方法（保持兼容性） ====================

    /**
     * 检测第一页的倾斜角度
     */
    public double detectSkewAngleFromFirstPage(MultipartFile file) {
        Path tempPath = null;
        PDDocument document = null;

        try {
            tempPath = uploadPath.resolve("temp_detect_" + UUID.randomUUID() + ".pdf");
            file.transferTo(tempPath.toFile());

            document = PDDocument.load(tempPath.toFile());
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage image = renderer.renderImageWithDPI(0, renderDpi);

            PageData pageData = new PageData(0, image, document.getPage(0).getMediaBox());
            PageAngleResult result = detectPageSkewAngle(pageData);

            image.flush();
            return result.getAngle();

        } catch (Exception e) {
            System.err.println("检测失败: " + e.getMessage());
            return 0.0;
        } finally {
            closeResource(document);
            deleteFile(tempPath);
        }
    }

    /**
     * 矫正PDF倾斜（简化版API）
     */
    public String correctPdfSkew(MultipartFile file) throws Exception {
        return correctPdfSkewWithAngle(file).getFileName();
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
        return resource;
    }

    /**
     * 关闭线程池
     */
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
    }

}