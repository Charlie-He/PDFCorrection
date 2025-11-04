package com.example.pdfcorrection.model;

import java.util.List;

/**
 * 矫正结果
 * 包含输出文件名和各页面的角度信息
 */
public class CorrectionResult {
    private final String fileName;
    private final List<Double> pageAngles;

    public CorrectionResult(String fileName, List<Double> pageAngles, long totalTime) {
        this.fileName = fileName;
        this.pageAngles = pageAngles;
    }

    public String getFileName() {
        return fileName;
    }

    public List<Double> getPageAngles() {
        return pageAngles;
    }

    /**
     * 获取第一页的角度（向后兼容）
     */
    public double getAngle() {
        return pageAngles.isEmpty() ? 0.0 : pageAngles.get(0);
    }
}