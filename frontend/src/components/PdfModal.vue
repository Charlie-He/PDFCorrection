<template>
  <div v-if="visible" class="pdf-modal" @click="handleOverlayClick">
    <div class="pdf-modal__content" :class="contentClass" @click.stop>
      <div class="pdf-modal__header">
        <h2>{{ title }}</h2>
        <button 
          class="pdf-modal__close"
          @click="close"
          aria-label="关闭"
        >
          ✕
        </button>
      </div>
      
      <div class="pdf-modal__body">
        <slot></slot>
      </div>
      
      <div v-if="$slots.footer" class="pdf-modal__footer">
        <slot name="footer"></slot>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PdfModal',
  props: {
    visible: {
      type: Boolean,
      default: false
    },
    title: {
      type: String,
      default: ''
    },
    closable: {
      type: Boolean,
      default: true
    },
    closeOnClickOverlay: {
      type: Boolean,
      default: true
    },
    contentClass: {
      type: String,
      default: ''
    }
  },
  emits: ['close', 'update:visible'],
  methods: {
    close() {
      if (this.closable) {
        this.$emit('close');
        this.$emit('update:visible', false);
      }
    },
    
    handleOverlayClick() {
      if (this.closeOnClickOverlay) {
        this.close();
      }
    }
  },
  
  watch: {
    visible(newVal) {
      if (newVal) {
        // 防止背景滚动
        document.body.style.overflow = 'hidden';
      } else {
        document.body.style.overflow = '';
      }
    }
  },
  
  beforeUnmount() {
    // 组件销毁前恢复滚动
    document.body.style.overflow = '';
  }
};
</script>

<style scoped>
.pdf-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
}

.pdf-modal__content {
  background: white;
  border-radius: 12px;
  width: 95vw;
  max-height: 90vh;
  overflow: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
}

.pdf-modal__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 30px;
  border-bottom: 1px solid #eee;
  position: sticky;
  top: 0;
  background: white;
  z-index: 10;
}

.pdf-modal__header h2 {
  margin: 0;
  color: #333;
  font-size: 1.5rem;
}

.pdf-modal__close {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: none;
  background: #f3f4f6;
  color: #666;
  cursor: pointer;
  font-size: 1.5rem;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.pdf-modal__close:hover {
  background: #e5e7eb;
  color: #333;
}

.pdf-modal__body {
  padding: 30px;
  flex: 1;
  overflow: auto;
}

.pdf-modal__footer {
  padding: 20px 30px;
  border-top: 1px solid #eee;
  display: flex;
  justify-content: flex-end;
  gap: 15px;
  position: sticky;
  bottom: 0;
  background: white;
  z-index: 10;
}

@media (max-width: 768px) {
  .pdf-modal__content {
    width: 100%;
    height: 100%;
    border-radius: 0;
  }
  
  .pdf-modal__header,
  .pdf-modal__body,
  .pdf-modal__footer {
    padding: 15px;
  }
  
  .pdf-modal__header h2 {
    font-size: 1.25rem;
  }
}
</style>