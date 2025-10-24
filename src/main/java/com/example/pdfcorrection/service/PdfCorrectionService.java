package com.example.pdfcorrection.service;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.*;
import org.opencv.core.Point;
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
 * PdfCorrectionService - 最终版（包含透视校正）
 *
 * 说明：
 * - 保持类名和外部 API 不变
 * - 保留并行处理（多线程）机制
 * - 在旋转矫正之后增加透视（warpPerspective）校正，将平行四边形/梯形拉正为矩形
 */
@Service
public class PdfCorrectionService {

    @Value("${file.upload-dir:uploads}")
    private String uploadDir;

    @Value("${pdf.correction.dpi:150}")
    private int renderDpi;

    // 小于该角度视为无需矫正（单位：度）
    @Value("${pdf.correction.min-angle:0.1}")
    private double minCorrectionAngle;

    // 透视校正的最小轮廓占比阈值（相对于图像面积），低于这个会被忽略
    @Value("${pdf.correction.perspective.min-area-ratio:0.2}")
    private double minPerspectiveAreaRatio;

    private Path uploadPath;
    private final ExecutorService executorService = Executors.newFixedThreadPool(
            Math.max(2, Runtime.getRuntime().availableProcessors() - 1));

    @PostConstruct
    public void init() {
        // Load OpenCV native lib (requires nu.pattern:opencv or similar)
        nu.pattern.OpenCV.loadLocally();
        uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
        try {
            Files.createDirectories(uploadPath);
            cleanupTempFiles();
        } catch (Exception e) {
            throw new RuntimeException("无法创建上传目录", e);
        }
    }

    @Scheduled(fixedRate = 3600000)
    public void scheduledCleanup() {
        cleanupTempFiles();
    }

    private void cleanupTempFiles() {
        try {
            Files.list(uploadPath)
                    .filter(path -> {
                        String n = path.getFileName().toString();
                        return n.startsWith("temp_") || n.startsWith("temp_page_") || n.startsWith("temp_detect_");
                    })
                    .filter(path -> {
                        try {
                            return System.currentTimeMillis() - Files.getLastModifiedTime(path).toMillis() > 3600000;
                        } catch (IOException e) {
                            return false;
                        }
                    })
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (Exception ignored) {
                        }
                    });
        } catch (Exception e) {
            System.err.println("清理临时文件失败: " + e.getMessage());
        }
    }

    /**
     * 主流程：PDF 矫正并保存
     */
    public CorrectionResult correctPdfSkewWithAngle(MultipartFile file) throws Exception {
        Path tempInputPath = null;
        Path outputPath = null;
        PDDocument document = null;
        PDDocument correctedDoc = null;

        long startTime = System.currentTimeMillis();

        try {
            String originalFileName = file.getOriginalFilename();
            String baseName = originalFileName != null && originalFileName.contains(".")
                    ? originalFileName.substring(0, originalFileName.lastIndexOf('.'))
                    : (originalFileName == null ? "output" : originalFileName);
            String correctedFileName = baseName + "_corrected_" + UUID.randomUUID().toString() + ".pdf";

            tempInputPath = uploadPath.resolve("temp_" + UUID.randomUUID().toString() + ".pdf");
            outputPath = uploadPath.resolve(correctedFileName);
            file.transferTo(tempInputPath.toFile());

            document = PDDocument.load(tempInputPath.toFile());
            correctedDoc = new PDDocument();
            int numberOfPages = document.getNumberOfPages();

            System.out.println("开始处理 " + numberOfPages + " 页PDF...");

            // 渲染每页为 BufferedImage
            PDFRenderer renderer = new PDFRenderer(document);
            List<PageImageData> pageImages = new ArrayList<>();
            List<PDRectangle> originalSizes = new ArrayList<>();

            for (int i = 0; i < numberOfPages; i++) {
                PDPage page = document.getPage(i);
                PDRectangle pageSize = page.getMediaBox();
                originalSizes.add(pageSize);

                BufferedImage image = renderer.renderImageWithDPI(i, renderDpi);
                pageImages.add(new PageImageData(i, image));
            }
            System.out.println("渲染完成，用时: " + (System.currentTimeMillis() - startTime) + "ms");

            // 并行检测角度
            long detectStart = System.currentTimeMillis();
            List<CompletableFuture<PageAngle>> angleFutures = pageImages.stream()
                    .map(pageData -> CompletableFuture.supplyAsync(() -> {
                        Mat mat = null;
                        try {
                            mat = bufferedImageToMat(pageData.getImage());
                            double angle = detectSkewAngleFast(mat);
                            System.out.println(String.format("第%d页检测角度: %.3f度", pageData.getIndex() + 1, angle));
                            return new PageAngle(pageData.getIndex(), angle);
                        } finally {
                            if (mat != null && !mat.empty()) mat.release();
                        }
                    }, executorService))
                    .collect(Collectors.toList());

            List<PageAngle> pageAngles = angleFutures.stream()
                    .map(CompletableFuture::join)
                    .sorted(Comparator.comparingInt(PageAngle::getIndex))
                    .collect(Collectors.toList());

            System.out.println("角度检测完成，用时: " + (System.currentTimeMillis() - detectStart) + "ms");

            // 并行执行矫正（旋转 + 透视）
            long correctStart = System.currentTimeMillis();
            List<CompletableFuture<CorrectedPage>> correctionFutures = new ArrayList<>();
            for (int i = 0; i < pageImages.size(); i++) {
                PageImageData pageData = pageImages.get(i);
                double angle = pageAngles.get(i).getAngle();

                CompletableFuture<CorrectedPage> future = CompletableFuture.supplyAsync(() -> {
                    try {
                        BufferedImage corrected = correctImageSkewAndPerspective(pageData.getImage(), angle);
                        return new CorrectedPage(pageData.getIndex(), corrected);
                    } finally {
                        pageData.getImage().flush();
                    }
                }, executorService);
                correctionFutures.add(future);
            }

            List<CorrectedPage> correctedPages = correctionFutures.stream()
                    .map(CompletableFuture::join)
                    .sorted(Comparator.comparingInt(CorrectedPage::getIndex))
                    .collect(Collectors.toList());

            System.out.println("图像矫正完成，用时: " + (System.currentTimeMillis() - correctStart) + "ms");

            // 写回 PDF（按原始页面大小缩放）
            long writeStart = System.currentTimeMillis();
            for (int i = 0; i < correctedPages.size(); i++) {
                CorrectedPage correctedPage = correctedPages.get(i);
                PDRectangle originalSize = originalSizes.get(i);
                writeCorrectedPageToPdf(correctedDoc, correctedPage, originalSize);
            }
            correctedDoc.save(outputPath.toFile());

            System.out.println("写入PDF完成，用时: " + (System.currentTimeMillis() - writeStart) + "ms");
            System.out.println("总用时: " + (System.currentTimeMillis() - startTime) + "ms");

            List<Double> angles = pageAngles.stream()
                    .map(PageAngle::getAngle)
                    .collect(Collectors.toList());

            return new CorrectionResult(correctedFileName, angles);

        } finally {
            closeResource(document);
            closeResource(correctedDoc);
            deleteFile(tempInputPath);
        }
    }

    /**
     * 检测角度：Hough + 投影融合（返回：正值 => 应逆时针旋转该角度）
     */
    private double detectSkewAngleFast(Mat srcColor) {
        Mat src = new Mat();
        Mat gray = new Mat();
        Mat binary = new Mat();

        try {
            if (srcColor.channels() == 3) {
                Imgproc.cvtColor(srcColor, src, Imgproc.COLOR_BGR2GRAY);
            } else {
                src = srcColor.clone();
            }

            int maxWidth = 1200;
            double scale = Math.min(1.0, (double) maxWidth / Math.max(1, src.cols()));
            Mat resized = new Mat();
            if (scale < 1.0) {
                Imgproc.resize(src, resized, new Size(src.cols() * scale, src.rows() * scale));
            } else {
                resized = src.clone();
            }

            Imgproc.GaussianBlur(resized, gray, new Size(3, 3), 0);
            Imgproc.adaptiveThreshold(gray, binary, 255,
                    Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, 9);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 1));
            Imgproc.dilate(binary, binary, kernel);
            kernel.release();

            Mat edges = new Mat();
            Imgproc.Canny(binary, edges, 50, 150);

            Mat lines = new Mat();
            Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, 40,
                    Math.max(30, resized.cols() / 8), 20);

            List<Double> candidateAngles = new ArrayList<>();

            if (lines.rows() > 0) {
                int maxInspect = Math.min(lines.rows(), 300);
                for (int i = 0; i < maxInspect; i++) {
                    double[] l = lines.get(i, 0);
                    double dx = l[2] - l[0];
                    double dy = l[3] - l[1];
                    double length = Math.hypot(dx, dy);
                    if (length < resized.cols() * 0.08) continue;
                    double angle = Math.toDegrees(Math.atan2(dy, dx));
                    while (angle > 90) angle -= 180;
                    while (angle < -90) angle += 180;
                    if (angle > 45) angle -= 90;
                    if (angle < -45) angle += 90;
                    if (Math.abs(angle) <= 45) candidateAngles.add(angle);
                }
            }

            double angleFromProjection = 0.0;
            if (candidateAngles.isEmpty()) {
                angleFromProjection = detectByFastProjection(binary);
            }

            double medianAngle = 0.0;
            if (!candidateAngles.isEmpty()) {
                Collections.sort(candidateAngles);
                medianAngle = filterOutliers(candidateAngles);
            }

            double initialAngle = candidateAngles.isEmpty() ? angleFromProjection : medianAngle;

            if (!candidateAngles.isEmpty() && Math.abs(initialAngle) < 0.5) {
                double proj = detectByFastProjection(binary);
                if (Math.abs(proj) > Math.abs(initialAngle) + 0.3) {
                    initialAngle = proj;
                }
            }

            if (Math.abs(initialAngle) < minCorrectionAngle) {
                resized.release();
                edges.release();
                lines.release();
                binary.release();
                gray.release();
                src.release();
                return 0.0;
            }

            double refined = refineAngleByProjection(binary, initialAngle);

            double signed = determineBestSignForAngle(resized, binary, refined);

            if (Math.abs(signed) < minCorrectionAngle) {
                resized.release();
                edges.release();
                lines.release();
                binary.release();
                gray.release();
                src.release();
                return 0.0;
            }

            resized.release();
            edges.release();
            lines.release();
            binary.release();
            gray.release();
            src.release();

            return signed;

        } catch (Exception e) {
            System.err.println("detectSkewAngleFast 出错: " + e.getMessage());
            if (!src.empty()) src.release();
            if (!gray.empty()) gray.release();
            if (!binary.empty()) binary.release();
            return 0.0;
        }
    }

    private double detectByFastProjection(Mat binary) {
        Mat small = new Mat();
        try {
            int w = Math.max(2, binary.cols() / 4);
            int h = Math.max(2, binary.rows() / 4);
            Imgproc.resize(binary, small, new Size(w, h));

            double bestAngle = 0;
            double maxScore = Double.NEGATIVE_INFINITY;

            for (int a = -15; a <= 15; a++) {
                Mat rotated = fastRotate(small, a);
                double score = calculateColumnProjectionScore(rotated);
                if (score > maxScore) {
                    maxScore = score;
                    bestAngle = a;
                }
                rotated.release();
            }
            return bestAngle;
        } catch (Exception e) {
            return 0.0;
        } finally {
            if (!small.empty()) small.release();
        }
    }

    private double refineAngleByProjection(Mat binary, double initialAngle) {
        Mat small = new Mat();
        try {
            int w = Math.max(2, binary.cols() / 2);
            int h = Math.max(2, binary.rows() / 2);
            Imgproc.resize(binary, small, new Size(w, h));

            double best = initialAngle;
            double bestScore = Double.NEGATIVE_INFINITY;
            double start = initialAngle - 1.5;
            double end = initialAngle + 1.5;

            for (double a = start; a <= end; a += 0.1) {
                Mat rotated = fastRotate(small, a);
                double score = calculateColumnProjectionScore(rotated);
                if (score > bestScore) {
                    bestScore = score;
                    best = a;
                }
                rotated.release();
            }
            return best;
        } catch (Exception e) {
            return initialAngle;
        } finally {
            if (!small.empty()) small.release();
        }
    }

    private double determineBestSignForAngle(Mat resizedGray, Mat binaryForProjection, double angle) {
        try {
            if (Math.abs(angle) < 1e-6) return 0.0;

            Mat small = new Mat();
            Imgproc.resize(binaryForProjection, small, new Size(Math.max(2, binaryForProjection.cols() / 3),
                    Math.max(2, binaryForProjection.rows() / 3)));

            Mat rPos = fastRotate(small, angle);
            Mat rNeg = fastRotate(small, -angle);
            double scorePos = calculateColumnProjectionScore(rPos);
            double scoreNeg = calculateColumnProjectionScore(rNeg);

            rPos.release();
            rNeg.release();
            small.release();

            return scorePos >= scoreNeg ? angle : -angle;
        } catch (Exception e) {
            return angle;
        }
    }

    private double calculateColumnProjectionScore(Mat mat) {
        int cols = mat.cols();
        int rows = mat.rows();
        if (cols <= 0 || rows <= 0) return Double.NEGATIVE_INFINITY;

        int step = Math.max(1, cols / 80);
        double sum = 0.0;
        double sumSq = 0.0;
        int count = 0;

        for (int c = 0; c < cols; c += step) {
            double colSum = 0;
            for (int r = 0; r < rows; r++) {
                double[] v = mat.get(r, c);
                colSum += v == null ? 0 : v[0];
            }
            sum += colSum;
            sumSq += colSum * colSum;
            count++;
        }
        double mean = sum / count;
        double variance = (sumSq / count) - (mean * mean);
        return variance;
    }

    private Mat fastRotate(Mat src, double angle) {
        Point center = new Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat rot = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Mat dst = new Mat();
        Imgproc.warpAffine(src, dst, rot, src.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
        rot.release();
        return dst;
    }

    private double filterOutliers(List<Double> sortedAngles) {
        if (sortedAngles == null || sortedAngles.isEmpty()) return 0.0;
        if (sortedAngles.size() < 5) {
            return sortedAngles.get(sortedAngles.size() / 2);
        }

        Collections.sort(sortedAngles);

        int n = sortedAngles.size();
        int q1Idx = n / 4;
        int q3Idx = n * 3 / 4;
        double q1 = sortedAngles.get(q1Idx);
        double q3 = sortedAngles.get(q3Idx);
        double iqr = q3 - q1;

        double lower = q1 - 1.5 * iqr;
        double upper = q3 + 1.5 * iqr;

        List<Double> filtered = new ArrayList<>();
        for (double a : sortedAngles) {
            if (a >= lower && a <= upper) filtered.add(a);
        }

        if (filtered.isEmpty()) {
            return sortedAngles.get(n / 2);
        }

        return filtered.get(filtered.size() / 2);
    }

    /**
     * 旋转 + 透视校正主函数
     */
    private BufferedImage correctImageSkewAndPerspective(BufferedImage image, double angle) {
        if (image == null) return null;
        if (Math.abs(angle) < minCorrectionAngle) {
            // 仍可能需要透视校正，即使角度很小 -> 仍尝试透视检测（谨慎）
            Mat mat = bufferedImageToMat(image);
            try {
                Mat maybe = fixPerspectiveIfNeeded(mat);
                return matToBufferedImage(maybe);
            } finally {
                if (!mat.empty()) mat.release();
            }
        }

        Mat mat = null;
        Mat rotated = null;
        try {
            mat = bufferedImageToMat(image);

            // 计算旋转后新尺寸（以留足边界）
            double radians = Math.toRadians(Math.abs(angle));
            int newWidth = (int) Math.ceil(mat.cols() * Math.cos(radians) + mat.rows() * Math.sin(radians));
            int newHeight = (int) Math.ceil(mat.cols() * Math.sin(radians) + mat.rows() * Math.cos(radians));
            newWidth = Math.max(mat.cols(), newWidth);
            newHeight = Math.max(mat.rows(), newHeight);

            // 使用检测得到的角度符号和反向作为备选，选使投影方差更大的结果
            Point center = new Point(mat.cols() / 2.0, mat.rows() / 2.0);

            // 正向旋转（使用检测返回的角度：正表示逆时针）
            Mat rotPosMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);
            double[] m0 = rotPosMat.get(0, 2);
            double[] m1 = rotPosMat.get(1, 2);
            double tx = m0[0] + (newWidth - mat.cols()) / 2.0;
            double ty = m1[0] + (newHeight - mat.rows()) / 2.0;
            rotPosMat.put(0, 2, tx);
            rotPosMat.put(1, 2, ty);
            Mat rotatedPos = new Mat();
            Imgproc.warpAffine(mat, rotatedPos, rotPosMat, new Size(newWidth, newHeight),
                    Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));
            rotPosMat.release();

            // 反向旋转
            Mat rotNegMat = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
            double[] n0 = rotNegMat.get(0, 2);
            double[] n1 = rotNegMat.get(1, 2);
            double ntx = n0[0] + (newWidth - mat.cols()) / 2.0;
            double nty = n1[0] + (newHeight - mat.rows()) / 2.0;
            rotNegMat.put(0, 2, ntx);
            rotNegMat.put(1, 2, nty);
            Mat rotatedNeg = new Mat();
            Imgproc.warpAffine(mat, rotatedNeg, rotNegMat, new Size(newWidth, newHeight),
                    Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));
            rotNegMat.release();

            // 比较两者得分
            double scorePos = calculateColumnProjectionScore(convertToGrayIfNeeded(rotatedPos));
            double scoreNeg = calculateColumnProjectionScore(convertToGrayIfNeeded(rotatedNeg));

            rotated = scorePos >= scoreNeg ? rotatedPos : rotatedNeg;
            // 释放未选中的 Mat
            if (rotated != rotatedPos) rotatedPos.release();
            if (rotated != rotatedNeg) rotatedNeg.release();

            // 透视校正（如果检测到明显的页面边界）
            Mat finalMat = fixPerspectiveIfNeeded(rotated);

            BufferedImage out = matToBufferedImage(finalMat);

            finalMat.release();
            return out;

        } catch (Exception e) {
            System.err.println("correctImageSkewAndPerspective 出错: " + e.getMessage());
            return image;
        } finally {
            if (mat != null && !mat.empty()) mat.release();
            if (rotated != null && !rotated.empty()) rotated.release();
        }
    }

    /**
     * 尝试检测页面四边（最大四边形轮廓），若检测到则进行透视变换将其矫正为矩形
     * 若未检测到符合条件的轮廓，则原样返回 src
     */
    private Mat fixPerspectiveIfNeeded(Mat src) {
        Mat gray = new Mat();
        Mat blurred = new Mat();
        Mat edged = new Mat();
        Mat srcCopy = src.clone();

        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);
            Imgproc.Canny(blurred, edged, 30, 150);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            double imgArea = src.size().area();
            double maxArea = 0;
            MatOfPoint2f bestQuad = null;

            for (MatOfPoint c : contours) {
                double area = Imgproc.contourArea(c);
                if (area < imgArea * minPerspectiveAreaRatio) {
                    c.release();
                    continue;
                }
                MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
                double peri = Imgproc.arcLength(c2f, true);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true);

                if (approx.total() == 4 && area > maxArea) {
                    maxArea = area;
                    if (bestQuad != null) bestQuad.release();
                    bestQuad = approx;
                } else {
                    approx.release();
                }
                c2f.release();
                c.release();
            }
            hierarchy.release();

            if (bestQuad == null) {
                return src;
            }

            // 将四点排序为 tl, tr, br, bl
            Point[] pts = bestQuad.toArray();
            Point[] sorted = sortCorners(pts);

            // 计算目标矩形宽高
            double widthA = distance(sorted[2], sorted[3]);
            double widthB = distance(sorted[1], sorted[0]);
            double maxWidth = Math.max(widthA, widthB);

            double heightA = distance(sorted[1], sorted[2]);
            double heightB = distance(sorted[0], sorted[3]);
            double maxHeight = Math.max(heightA, heightB);

            // 如果尺寸非常小或为0，回退
            if (maxWidth < 10 || maxHeight < 10) {
                bestQuad.release();
                return src;
            }

            MatOfPoint2f srcPts = new MatOfPoint2f(sorted[0], sorted[1], sorted[2], sorted[3]);
            MatOfPoint2f dstPts = new MatOfPoint2f(
                    new Point(0, 0),
                    new Point(maxWidth - 1, 0),
                    new Point(maxWidth - 1, maxHeight - 1),
                    new Point(0, maxHeight - 1)
            );

            Mat transform = Imgproc.getPerspectiveTransform(srcPts, dstPts);
            Mat warped = new Mat();
            Imgproc.warpPerspective(src, warped, transform, new Size(maxWidth, maxHeight),
                    Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));

            // 释放资源
            bestQuad.release();
            srcPts.release();
            dstPts.release();
            transform.release();

            return warped;

        } catch (Exception e) {
            System.err.println("fixPerspectiveIfNeeded 出错: " + e.getMessage());
            return srcCopy;
        } finally {
            gray.release();
            blurred.release();
            edged.release();
            // srcCopy 在成功路径会被释放或返回后释放调用处，若不需要可释放
        }
    }

    /**
     * 对四个角点进行排序：返回 [tl, tr, br, bl]
     */
    private Point[] sortCorners(Point[] pts) {
        Point[] result = new Point[4];
        // 按 x+y 排序（tl 最小，br 最大）
        Arrays.sort(pts, Comparator.comparingDouble(p -> (p.y + p.x)));
        result[0] = pts[0]; // tentative tl
        result[2] = pts[3]; // tentative br

        // 剩下两个是 tr 与 bl，按 y-x 区分
        List<Point> mid = new ArrayList<>();
        mid.add(pts[1]);
        mid.add(pts[2]);
        mid.sort(Comparator.comparingDouble(p -> (p.y - p.x)));
        result[1] = mid.get(0); // tr
        result[3] = mid.get(1); // bl
        return result;
    }

    private double distance(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    private Mat convertToGrayIfNeeded(Mat src) {
        if (src.channels() == 1) return src;
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        return gray;
    }

    private void writeCorrectedPageToPdf(PDDocument doc, CorrectedPage page, PDRectangle originalPageSize) throws IOException {
        BufferedImage image = page.getImage();

        float originalWidth = originalPageSize.getWidth();
        float originalHeight = originalPageSize.getHeight();
        PDPage pdPage = new PDPage(new PDRectangle(originalWidth, originalHeight));
        doc.addPage(pdPage);

        Path tempImagePath = uploadPath.resolve("temp_page_" + UUID.randomUUID().toString() + ".png");

        try {
            ImageIO.write(image, "PNG", tempImagePath.toFile());
            PDImageXObject pdImage = PDImageXObject.createFromFile(tempImagePath.toString(), doc);

            try (PDPageContentStream contentStream = new PDPageContentStream(doc, pdPage)) {
                contentStream.drawImage(pdImage, 0, 0, originalWidth, originalHeight);
            }
        } finally {
            deleteFile(tempImagePath);
            image.flush();
        }
    }

    public double detectSkewAngleFromFirstPage(MultipartFile file) {
        Path tempPath = null;
        PDDocument document = null;
        Mat mat = null;

        try {
            tempPath = uploadPath.resolve("temp_detect_" + UUID.randomUUID().toString() + ".pdf");
            file.transferTo(tempPath.toFile());

            document = PDDocument.load(tempPath.toFile());
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage image = renderer.renderImageWithDPI(0, renderDpi);

            mat = bufferedImageToMat(image);
            double angle = detectSkewAngleFast(mat);

            image.flush();
            return angle;

        } catch (Exception e) {
            System.err.println("检测失败: " + e.getMessage());
            return 0.0;
        } finally {
            if (mat != null && !mat.empty()) mat.release();
            closeResource(document);
            deleteFile(tempPath);
        }
    }

    public String correctPdfSkew(MultipartFile file) throws Exception {
        return correctPdfSkewWithAngle(file).getFileName();
    }

    /**
     * BufferedImage -> Mat (BGR)
     */
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

    /**
     * Mat -> BufferedImage
     */
    private BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) return null;
        int type = mat.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        byte[] data = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, data);
        byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, targetPixels, 0, data.length);
        return image;
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
    }

    private void closeResource(AutoCloseable resource) {
        if (resource != null) {
            try {
                resource.close();
            } catch (Exception ignored) {
            }
        }
    }

    private void deleteFile(Path path) {
        if (path != null) {
            try {
                Files.deleteIfExists(path);
            } catch (Exception ignored) {
            }
        }
    }

    // 内部类
    public static class CorrectionResult {
        private final String fileName;
        private final List<Double> pageAngles;

        public CorrectionResult(String fileName, List<Double> pageAngles) {
            this.fileName = fileName;
            this.pageAngles = pageAngles;
        }

        public String getFileName() {
            return fileName;
        }

        public List<Double> getPageAngles() {
            return pageAngles;
        }

        public double getAngle() {
            return pageAngles.isEmpty() ? 0.0 : pageAngles.get(0);
        }
    }

    private static class PageAngle {
        private final int index;
        private final double angle;

        public PageAngle(int index, double angle) {
            this.index = index;
            this.angle = angle;
        }

        public int getIndex() {
            return index;
        }

        public double getAngle() {
            return angle;
        }
    }

    private static class CorrectedPage {
        private final int index;
        private final BufferedImage image;

        public CorrectedPage(int index, BufferedImage image) {
            this.index = index;
            this.image = image;
        }

        public int getIndex() {
            return index;
        }

        public BufferedImage getImage() {
            return image;
        }
    }

    private static class PageImageData {
        private final int index;
        private final BufferedImage image;

        public PageImageData(int index, BufferedImage image) {
            this.index = index;
            this.image = image;
        }

        public int getIndex() {
            return index;
        }

        public BufferedImage getImage() {
            return image;
        }
    }
}