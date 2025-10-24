package com.example.pdfcorrection.model;

/**
 * 页面角度检测结果
 * 包含页面索引和检测到的倾斜角度
 */
public class PageAngleResult {
    private final int pageIndex;
    private final double angle;

    public PageAngleResult(int pageIndex, double angle) {
        this.pageIndex = pageIndex;
        this.angle = angle;
    }

    public int getPageIndex() {
        return pageIndex;
    }

    public double getAngle() {
        return angle;
    }
}