<template>
  <div 
    class="pdf-upload"
    :class="{
      'pdf-upload--drag-over': isDragging,
      'pdf-upload--disabled': disabled
    }"
    @drop.prevent="handleDrop"
    @dragover.prevent="handleDragOver"
    @dragleave="handleDragLeave"
  >
    <input
      ref="fileInput"
      type="file"
      :accept="accept"
      :multiple="multiple"
      @change="handleFileSelect"
      class="pdf-upload__input"
    >

    <div v-if="!modelValue" class="pdf-upload__prompt">
      <slot name="prompt">
        <svg class="pdf-upload__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <p class="pdf-upload__text">拖拽文件到此处，或者</p>
        <PdfButton @click="triggerFileInput" variant="primary">
          选择文件
        </PdfButton>
      </slot>
    </div>

    <div v-else class="pdf-upload__file-info">
      <slot name="file-info" :file="modelValue" :remove-file="removeFile">
        <svg class="pdf-upload__file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        <div class="pdf-upload__file-details">
          <p class="pdf-upload__file-name">{{ modelValue.name }}</p>
          <p class="pdf-upload__file-size">{{ formatFileSize(modelValue.size) }}</p>
        </div>
        <button 
          v-if="!disabled"
          class="pdf-upload__remove"
          @click="removeFile"
          :disabled="processing"
          aria-label="移除文件"
        >
          ✕
        </button>
      </slot>
    </div>
  </div>
</template>

<script>
import PdfButton from './PdfButton.vue';

export default {
  name: 'PdfUpload',
  components: {
    PdfButton
  },
  props: {
    modelValue: {
      type: [File, null],
      default: null
    },
    accept: {
      type: String,
      default: '*'
    },
    multiple: {
      type: Boolean,
      default: false
    },
    disabled: {
      type: Boolean,
      default: false
    },
    processing: {
      type: Boolean,
      default: false
    }
  },
  emits: ['update:modelValue', 'file-select'],
  data() {
    return {
      isDragging: false
    };
  },
  methods: {
    triggerFileInput() {
      if (!this.disabled) {
        this.$refs.fileInput.click();
      }
    },

    handleFileSelect(event) {
      const file = event.target.files[0];
      if (file) {
        this.$emit('update:modelValue', file);
        this.$emit('file-select', file);
      }
      // 重置input的value，确保下次选择相同文件时也能触发change事件
      event.target.value = '';
    },

    handleDrop(event) {
      if (this.disabled) return;
      
      this.isDragging = false;
      const file = event.dataTransfer.files[0];
      if (file) {
        this.$emit('update:modelValue', file);
        this.$emit('file-select', file);
      }
    },

    handleDragOver() {
      if (!this.disabled) {
        this.isDragging = true;
      }
    },

    handleDragLeave() {
      this.isDragging = false;
    },

    removeFile() {
      if (!this.disabled && !this.processing) {
        this.$refs.fileInput.value = '';
        this.$emit('update:modelValue', null);
      }
    },

    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
  }
};
</script>

<style scoped>
.pdf-upload {
  border: 2px dashed #ddd;
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  transition: all 0.3s ease;
  background: #fafafa;
  position: relative;
}

.pdf-upload--drag-over {
  border-color: #667eea;
  background: #f0f4ff;
}

.pdf-upload--disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.pdf-upload__input {
  display: none;
}

.pdf-upload__icon {
  width: 64px;
  height: 64px;
  color: #667eea;
  stroke-width: 2;
  margin: 0 auto 20px;
}

.pdf-upload__text {
  color: #666;
  font-size: 1rem;
  margin-bottom: 20px;
}

.pdf-upload__file-info {
  display: flex;
  align-items: center;
  gap: 15px;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.pdf-upload__file-icon {
  width: 40px;
  height: 40px;
  color: #667eea;
  stroke-width: 2;
  flex-shrink: 0;
}

.pdf-upload__file-details {
  flex: 1;
  text-align: left;
}

.pdf-upload__file-name {
  font-weight: 600;
  color: #333;
  margin-bottom: 5px;
  word-break: break-all;
}

.pdf-upload__file-size {
  color: #999;
  font-size: 0.875rem;
}

.pdf-upload__remove {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: none;
  background: #ff4444;
  color: white;
  cursor: pointer;
  font-size: 1.2rem;
  transition: all 0.3s;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.pdf-upload__remove:hover:not(:disabled) {
  background: #cc0000;
  transform: scale(1.1);
}

.pdf-upload__remove:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>