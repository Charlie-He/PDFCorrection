<template>
  <button 
    class="pdf-button" 
    :class="[
      `pdf-button--${variant}`, 
      `pdf-button--${size}`,
      {
        'pdf-button--full-width': fullWidth,
        'pdf-button--loading': loading,
        'pdf-button--disabled': disabled
      }
    ]"
    :disabled="disabled || loading"
    @click="handleClick"
  >
    <span v-if="loading" class="pdf-button__spinner"></span>
    <slot v-else></slot>
  </button>
</template>

<script>
export default {
  name: 'PdfButton',
  props: {
    variant: {
      type: String,
      default: 'primary',
      validator: (value) => [
        'primary', 
        'secondary', 
        'success', 
        'danger', 
        'outline-primary',
        'outline-secondary'
      ].includes(value)
    },
    size: {
      type: String,
      default: 'medium',
      validator: (value) => ['small', 'medium', 'large'].includes(value)
    },
    fullWidth: {
      type: Boolean,
      default: false
    },
    loading: {
      type: Boolean,
      default: false
    },
    disabled: {
      type: Boolean,
      default: false
    }
  },
  emits: ['click'],
  methods: {
    handleClick(event) {
      if (!this.disabled && !this.loading) {
        this.$emit('click', event);
      }
    }
  }
};
</script>

<style scoped>
.pdf-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  outline: none;
  gap: 8px;
}

.pdf-button:focus-visible {
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.4);
}

.pdf-button--full-width {
  width: 100%;
}

.pdf-button--small {
  padding: 8px 16px;
  font-size: 0.875rem;
}

.pdf-button--medium {
  padding: 12px 24px;
  font-size: 1rem;
}

.pdf-button--large {
  padding: 16px 32px;
  font-size: 1.125rem;
}

/* Primary Button */
.pdf-button--primary {
  background: #667eea;
  color: white;
}

.pdf-button--primary:hover:not(.pdf-button--disabled):not(.pdf-button--loading) {
  background: #5568d3;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* Secondary Button */
.pdf-button--secondary {
  background: #6b7280;
  color: white;
}

.pdf-button--secondary:hover:not(.pdf-button--disabled):not(.pdf-button--loading) {
  background: #4b5563;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(107, 114, 128, 0.4);
}

/* Success Button */
.pdf-button--success {
  background: #10b981;
  color: white;
}

.pdf-button--success:hover:not(.pdf-button--disabled):not(.pdf-button--loading) {
  background: #059669;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}

/* Danger Button */
.pdf-button--danger {
  background: #ef4444;
  color: white;
}

.pdf-button--danger:hover:not(.pdf-button--disabled):not(.pdf-button--loading) {
  background: #dc2626;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
}

/* Outline Primary Button */
.pdf-button--outline-primary {
  background: transparent;
  color: #667eea;
  border: 2px solid #667eea;
}

.pdf-button--outline-primary:hover:not(.pdf-button--disabled):not(.pdf-button--loading) {
  background: #667eea;
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* Outline Secondary Button */
.pdf-button--outline-secondary {
  background: transparent;
  color: #6b7280;
  border: 2px solid #6b7280;
}

.pdf-button--outline-secondary:hover:not(.pdf-button--disabled):not(.pdf-button--loading) {
  background: #6b7280;
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(107, 114, 128, 0.4);
}

/* Disabled State */
.pdf-button--disabled,
.pdf-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

/* Loading State */
.pdf-button--loading {
  cursor: wait;
}

.pdf-button__spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* Full Width Adjustment */
.pdf-button--full-width.pdf-button--large {
  padding-left: 16px;
  padding-right: 16px;
}
</style>