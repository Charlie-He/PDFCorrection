<template>
  <div id="app">
    <div class="container">
      <h1 class="title">📄 PDF 页面倾斜校正系统</h1>

      <div class="upload-section">
        <PdfUpload
          v-model="selectedFile"
          accept=".pdf"
          :processing="isProcessing"
          @file-select="onFileSelect"
        />

        <div v-if="selectedFile && !isProcessing && !correctedFile" class="action-buttons">
          <PdfButton variant="outline-primary" @click="showPreview" fullWidth>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="20" height="20">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
            预览原文件
          </PdfButton>
          <PdfButton variant="success" @click="uploadAndCorrect" :loading="isProcessing" fullWidth>
            开始校正
          </PdfButton>
        </div>
      </div>

      <PdfProgress
        :show="isProcessing"
        type="bar"
        :progress="progressValue"
        message="正在处理中，请稍候..."
      />

      <div v-if="errorMessage" class="error-message">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        {{ errorMessage }}
      </div>

      <div v-if="correctedFile" class="success-section">
        <div class="success-message">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
          </svg>
          <span>校正完成！</span>
        </div>
        
        <div class="correction-info">
          <div v-if="pageAngles && pageAngles.length > 0" class="page-angles">
            <h3>各页面倾斜角度：</h3>
            <ul>
              <li v-for="(angle, index) in pageAngles" :key="index">
                第 {{ index + 1 }} 页: {{ angle.toFixed(2) }}°
              </li>
            </ul>
          </div>
          <p v-else>检测到的倾斜角度: {{ detectedAngle.toFixed(2) }}°</p>
          <p>处理耗时: {{ processingTime.toFixed(2) }}秒</p>
        </div>

        <div class="action-buttons">
          <PdfButton variant="outline-primary" @click="showCompareView" fullWidth>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="20" height="20">
              <rect x="3" y="3" width="7" height="7"></rect>
              <rect x="14" y="3" width="7" height="7"></rect>
              <rect x="14" y="14" width="7" height="7"></rect>
              <rect x="3" y="14" width="7" height="7"></rect>
            </svg>
            对比预览
          </PdfButton>
          <PdfButton variant="primary" @click="downloadFile" fullWidth>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="20" height="20">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            下载校正后的PDF
          </PdfButton>
        </div>

        <PdfButton variant="secondary" @click="reset" fullWidth class="reset-button">
          处理新文件
        </PdfButton>
      </div>
    </div>

    <PdfModal v-model:visible="showPreviewModal" :title="previewTitle" content-class="preview-modal">
      <div v-if="compareMode" class="compare-container">
        <div class="preview-panel">
          <h3>原始文件</h3>
          <div class="pdf-viewer">
            <iframe :src="originalPdfUrl" v-if="originalPdfUrl"></iframe>
            <div v-else>加载中...</div>
          </div>
        </div>

        <div class="preview-panel">
          <h3>校正后文件</h3>
          <div class="pdf-viewer">
            <iframe :src="correctedPdfUrl" v-if="correctedPdfUrl"></iframe>
            <div v-else>加载中...</div>
          </div>
        </div>
      </div>

      <div v-else class="single-preview">
        <div class="pdf-viewer">
          <iframe :src="previewPdfUrl" v-if="previewPdfUrl"></iframe>
          <div v-else>加载中...</div>
        </div>
      </div>
    </PdfModal>
  </div>
</template>

<script>
import axios from 'axios';
import PdfUpload from './components/PdfUpload.vue';
import PdfButton from './components/PdfButton.vue';
import PdfProgress from './components/PdfProgress.vue';
import PdfModal from './components/PdfModal.vue';

export default {
  name: 'App',
  components: {
    PdfUpload,
    PdfButton,
    PdfProgress,
    PdfModal
  },
  data() {
    return {
      selectedFile: null,
      isProcessing: false,
      correctedFile: null,
      errorMessage: '',
      showPreviewModal: false,
      compareMode: false,
      previewTitle: '',
      originalPdfUrl: null,
      correctedPdfUrl: null,
      previewPdfUrl: null,
      detectedAngle: 0,
      pageAngles: [],
      processingTime: 0,
      startTime: 0,
      progressValue: 0,
      progressInterval: null
    };
  },
  methods: {
    onFileSelect() {
      this.errorMessage = '';
    },

    showPreview() {
      if (!this.selectedFile) {
        this.errorMessage = '未选择文件';
        return;
      }

      this.compareMode = false;
      this.previewTitle = '原始PDF预览';
      this.showPreviewModal = true;

      this.previewPdfUrl = URL.createObjectURL(this.selectedFile);
    },

    async showCompareView() {
      if (!this.selectedFile) {
        this.errorMessage = '未选择文件';
        return;
      }

      if (!this.correctedFile) {
        this.errorMessage = '请先处理文件再进行对比预览';
        return;
      }

      this.compareMode = true;
      this.previewTitle = '校正前后对比';
      this.showPreviewModal = true;

      this.originalPdfUrl = URL.createObjectURL(this.selectedFile);

      try {
        const response = await axios.get(
          `http://localhost:8080/api/pdf/download/${this.correctedFile}`,
          { responseType: 'blob' }
        );
        
        this.correctedPdfUrl = URL.createObjectURL(response.data);
      } catch (error) {
        console.error('加载校正文件失败:', error);
        this.errorMessage = '加载校正文件失败: ' + (error.message || '未知错误');
      }
    },

    closePreview() {
      this.showPreviewModal = false;
      
      if (this.originalPdfUrl) {
        URL.revokeObjectURL(this.originalPdfUrl);
        this.originalPdfUrl = null;
      }
      
      if (this.correctedPdfUrl) {
        URL.revokeObjectURL(this.correctedPdfUrl);
        this.correctedPdfUrl = null;
      }
      
      if (this.previewPdfUrl) {
        URL.revokeObjectURL(this.previewPdfUrl);
        this.previewPdfUrl = null;
      }
    },

    async uploadAndCorrect() {
      if (!this.selectedFile) {
        this.errorMessage = '未选择文件';
        return;
      }

      this.isProcessing = true;
      this.errorMessage = '';
      this.startTime = Date.now();
      this.progressValue = 0;
      
      // 模拟进度条增长
      this.progressInterval = setInterval(() => {
        if (this.progressValue < 90) {
          this.progressValue += 5;
        }
      }, 100);

      const formData = new FormData();
      formData.append('file', this.selectedFile);

      try {
        const response = await axios.post('http://localhost:8080/api/pdf/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        // 确保进度条达到100%
        this.progressValue = 100;
        
        if (this.progressInterval) {
          clearInterval(this.progressInterval);
          this.progressInterval = null;
        }

        if (response.data.success) {
          this.correctedFile = response.data.fileName;
          this.pageAngles = response.data.pageAngles || [];
          
          if (this.pageAngles && this.pageAngles.length > 0) {
            const sum = this.pageAngles.reduce((a, b) => a + b, 0);
            this.detectedAngle = sum / this.pageAngles.length;
          } else {
            this.detectedAngle = response.data.angle || 0;
          }
          
          this.processingTime = (Date.now() - this.startTime) / 1000;
        } else {
          this.errorMessage = response.data.message || '处理失败';
        }
      } catch (error) {
        console.error('文件上传失败:', error);
        if (this.progressInterval) {
          clearInterval(this.progressInterval);
          this.progressInterval = null;
        }
        
        if (error.response && error.response.data && error.response.data.message) {
          this.errorMessage = error.response.data.message;
        } else if (error.response) {
          this.errorMessage = '服务器处理失败: ' + (error.response.statusText || '未知错误');
        } else if (error.request) {
          this.errorMessage = '无法连接到服务器，请检查网络连接或服务器状态';
        } else {
          this.errorMessage = '上传失败: ' + (error.message || '未知错误');
        }
      } finally {
        this.isProcessing = false;
        if (this.progressInterval) {
          clearInterval(this.progressInterval);
          this.progressInterval = null;
          this.progressValue = 100;
        }
      }
    },

    async downloadFile() {
      if (!this.correctedFile) return;

      try {
        const response = await axios.get(
          `http://localhost:8080/api/pdf/download/${this.correctedFile}`,
          { responseType: 'blob' }
        );
        
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', this.correctedFile);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      } catch (error) {
        console.error('下载失败:', error);
        if (error.response && error.response.headers['error-message']) {
          this.errorMessage = '下载失败: ' + error.response.headers['error-message'];
        } else {
          this.errorMessage = '下载失败，请重试';
        }
      }
    },

    reset() {
      this.selectedFile = null;
      this.correctedFile = null;
      this.pageAngles = [];
      this.errorMessage = '';
      this.progressValue = 0;
    }
  },
  
  beforeUnmount() {
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
    }
  }
};
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 40px 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
}

.container {
  max-width: 800px;
  margin: 0 auto;
}

.title {
  text-align: center;
  color: white;
  font-size: 2rem;
  margin-bottom: 40px;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.upload-section {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.action-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
  margin-top: 30px;
}

@media (max-width: 768px) {
  .action-buttons {
    grid-template-columns: 1fr;
  }
  
  .title {
    font-size: 1.5rem;
    margin-bottom: 30px;
  }
  
  .upload-section {
    padding: 20px;
  }
}

.error-message {
  margin-top: 30px;
  padding: 15px;
  background: #fee;
  border: 1px solid #fcc;
  border-radius: 8px;
  color: #c33;
  display: flex;
  align-items: center;
  gap: 10px;
}

.error-message svg {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
  stroke-width: 2;
}

.success-section {
  margin-top: 30px;
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.success-message {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #10b981;
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 25px;
}

.success-message svg {
  width: 32px;
  height: 32px;
  stroke-width: 2;
}

.correction-info {
  text-align: center;
  margin: 20px 0;
  padding: 15px;
  background: #f0f9ff;
  border-radius: 8px;
  color: #0369a1;
  font-weight: 500;
}

.correction-info h3 {
  margin-bottom: 10px;
}

.correction-info ul {
  list-style-type: none;
  padding: 0;
  margin: 10px 0;
  text-align: left;
  display: inline-block;
}

.correction-info li {
  margin: 5px 0;
}

.correction-info p {
  margin: 5px 0;
}

.reset-button {
  margin-top: 20px;
}

/* 预览模态框样式 */
.preview-modal {
  width: 95vw;
  height: 90vh;
}

.compare-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  min-height: 800px;
}

.preview-panel h3 {
  text-align: center;
  color: #667eea;
  margin-bottom: 15px;
  font-size: 1.2rem;
}

.single-preview {
  padding: 10px;
}

.pdf-viewer {
  background: #f9fafb;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 800px;
  overflow: auto;
}

.pdf-viewer iframe {
  width: 100%;
  height: 800px;
  border: none;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border-radius: 4px;
}

@media (max-width: 768px) {
  .compare-container {
    grid-template-columns: 1fr;
  }
  
  .pdf-viewer {
    min-height: 500px;
    padding: 10px;
  }
  
  .pdf-viewer iframe {
    height: 500px;
  }
  
  .preview-modal {
    width: 100%;
    height: 100%;
    border-radius: 0;
  }
  
  .success-section,
  .upload-section {
    padding: 20px;
  }
}
</style>