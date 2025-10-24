package com.example.pdfcorrection.controller;

import com.example.pdfcorrection.model.CorrectionResult;
import com.example.pdfcorrection.service.PdfCorrectionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.net.URLEncoder;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/pdf")
@CrossOrigin(origins = "*")
public class PdfController {

    @Autowired
    private PdfCorrectionService pdfCorrectionService;

    @PostMapping("/upload")
    public ResponseEntity<?> uploadAndCorrectPdf(@RequestParam("file") MultipartFile file) {
        try {
            if (file.isEmpty()) {
                return ResponseEntity.badRequest().body(new UploadResponse(false, "文件不能为空", null, new ArrayList<>()));
            }

            if (!file.getOriginalFilename().toLowerCase().endsWith(".pdf")) {
                return ResponseEntity.badRequest().body(new UploadResponse(false, "只支持PDF文件", null, new ArrayList<>()));
            }

            // 一次性完成角度检测和PDF校正，避免重复使用MultipartFile
            CorrectionResult result = pdfCorrectionService.correctPdfSkewWithAngle(file);
            
            return ResponseEntity.ok()
                    .body(new UploadResponse(true, "PDF校正成功", result.getFileName(), result.getPageAngles()));

        } catch (Exception e) {
            e.printStackTrace();
            // 返回详细的错误信息给前端
            return ResponseEntity.internalServerError()
                    .body(new UploadResponse(false, "处理失败: " + e.getMessage(), null, new ArrayList<>()));
        }
    }

    @GetMapping("/download/{fileName}")
    public ResponseEntity<Resource> downloadCorrectedPdf(@PathVariable String fileName) {
        try {
            Resource resource = pdfCorrectionService.loadFileAsResource(fileName);

            String encodedFileName = URLEncoder.encode(resource.getFilename(), "UTF-8");
            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_PDF)
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            "attachment; filename*=UTF-8''" + encodedFileName)
                    .body(resource);

        } catch (Exception e) {
            e.printStackTrace();
            // 返回错误信息给前端
            return ResponseEntity.internalServerError()
                    .header("error-message", e.getMessage())
                    .build();
        }
    }

    static class UploadResponse {
        private boolean success;
        private String message;
        private String fileName;
        private List<Double> pageAngles;

        public UploadResponse(boolean success, String message, String fileName, List<Double> pageAngles) {
            this.success = success;
            this.message = message;
            this.fileName = fileName;
            this.pageAngles = pageAngles;
        }

        public boolean isSuccess() {
            return success;
        }

        public void setSuccess(boolean success) {
            this.success = success;
        }

        public String getMessage() {
            return message;
        }

        public void setMessage(String message) {
            this.message = message;
        }

        public String getFileName() {
            return fileName;
        }

        public void setFileName(String fileName) {
            this.fileName = fileName;
        }
        
        public List<Double> getPageAngles() {
            return pageAngles;
        }
        
        public void setPageAngles(List<Double> pageAngles) {
            this.pageAngles = pageAngles;
        }
    }
}