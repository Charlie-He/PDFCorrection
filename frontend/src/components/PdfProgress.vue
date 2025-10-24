<template>
  <div class="pdf-progress" v-if="show">
    <div class="pdf-progress__indicator">
      <div 
        class="pdf-progress__spinner" 
        v-if="type === 'spinner'"
      ></div>
      
      <div 
        class="pdf-progress__bar"
        v-else-if="type === 'bar'"
      >
        <div 
          class="pdf-progress__bar-fill"
          :style="{ width: `${progress}%` }"
        ></div>
      </div>
    </div>
    
    <p class="pdf-progress__message">{{ message }}</p>
  </div>
</template>

<script>
export default {
  name: 'PdfProgress',
  props: {
    show: {
      type: Boolean,
      default: false
    },
    type: {
      type: String,
      default: 'spinner', // 'spinner' or 'bar'
      validator: (value) => ['spinner', 'bar'].includes(value)
    },
    message: {
      type: String,
      default: '处理中...'
    },
    progress: {
      type: Number,
      default: 0,
      validator: (value) => value >= 0 && value <= 100
    }
  }
};
</script>

<style scoped>
.pdf-progress {
  margin: 30px 0;
  text-align: center;
  color: white;
}

.pdf-progress__indicator {
  margin-bottom: 20px;
}

.pdf-progress__spinner {
  width: 50px;
  height: 50px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  margin: 0 auto 20px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.pdf-progress__bar {
  width: 100%;
  height: 12px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  overflow: hidden;
  margin: 0 auto 20px;
  max-width: 400px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.pdf-progress__bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 6px;
  transition: width 0.3s ease;
}

.pdf-progress__message {
  font-size: 1.1rem;
  margin-bottom: 10px;
  font-weight: 500;
}
</style>