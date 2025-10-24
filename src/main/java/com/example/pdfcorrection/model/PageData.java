package com.example.pdfcorrection.model;

import org.apache.pdfbox.pdmodel.common.PDRectangle;

import java.awt.image.BufferedImage;

/**
 * 页面数据传输对象
 * 用于在PDF处理流程中传递页面相关信息
 */
public class PageData {
    private final int pageIndex;
    private final BufferedImage image;
    private final PDRectangle originalSize;

    public PageData(int pageIndex, BufferedImage image, PDRectangle originalSize) {
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