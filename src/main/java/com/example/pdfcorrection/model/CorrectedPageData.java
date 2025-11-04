package com.example.pdfcorrection.model;

import lombok.Getter;
import org.apache.pdfbox.pdmodel.common.PDRectangle;

import java.awt.image.BufferedImage;

/**
 * 矫正后的页面数据
 * 包含页面索引、矫正后的图像和原始页面尺寸
 */
@Getter
public class CorrectedPageData {
    private final int pageIndex;
    private final BufferedImage image;
    private final PDRectangle originalSize;
    private final boolean needsCorrection;


    public CorrectedPageData(int pageIndex, BufferedImage image, PDRectangle originalSize, boolean b) {
        this.pageIndex = pageIndex;
        this.image = image;
        this.originalSize = originalSize;
        this.needsCorrection = b;
    }

    public boolean needsCorrection() {
        return needsCorrection;
    }
}