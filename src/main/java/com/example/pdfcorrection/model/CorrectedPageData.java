package com.example.pdfcorrection.model;

import org.apache.pdfbox.pdmodel.common.PDRectangle;

import java.awt.image.BufferedImage;

/**
 * 矫正后的页面数据
 * 包含页面索引、矫正后的图像和原始页面尺寸
 */
public class CorrectedPageData {
    private final int pageIndex;
    private final BufferedImage image;
    private final PDRectangle originalSize;

    public CorrectedPageData(int pageIndex, BufferedImage image, PDRectangle originalSize) {
        this.pageIndex = pageIndex;
        this.image = image;
        this.originalSize = originalSize;
    }

    public int getPageIndex() {
        return pageIndex;
    }

    public BufferedImage getImage() {
        return image;
    }

    public PDRectangle getOriginalSize() {
        return originalSize;
    }
}