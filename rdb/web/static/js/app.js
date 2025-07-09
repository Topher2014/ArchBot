// Main JavaScript file for RDB web interface

// Global utilities
window.RDB = {
   // API helper functions
   api: {
       async get(url) {
           const response = await fetch(url);
           if (!response.ok) {
               throw new Error(`HTTP error! status: ${response.status}`);
           }
           return await response.json();
       },
       
       async post(url, data) {
           const response = await fetch(url, {
               method: 'POST',
               headers: {
                   'Content-Type': 'application/json',
               },
               body: JSON.stringify(data)
           });
           if (!response.ok) {
               throw new Error(`HTTP error! status: ${response.status}`);
           }
           return await response.json();
       },
       
       async delete(url) {
           const response = await fetch(url, {
               method: 'DELETE'
           });
           if (!response.ok) {
               throw new Error(`HTTP error! status: ${response.status}`);
           }
           return await response.json();
       }
   },
   
   // UI utilities
   ui: {
       showLoading(element) {
           if (typeof element === 'string') {
               element = document.getElementById(element);
           }
           if (element) {
               element.innerHTML = `
                   <div class="flex justify-center items-center py-4">
                       <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                       <span class="ml-2 text-gray-600">Loading...</span>
                   </div>
               `;
           }
       },
       
       showError(element, message) {
           if (typeof element === 'string') {
               element = document.getElementById(element);
           }
           if (element) {
               element.innerHTML = `
                   <div class="message-error">
                       <i class="fas fa-exclamation-triangle mr-2"></i>
                       ${message}
                   </div>
               `;
           }
       },
       
       showSuccess(element, message) {
           if (typeof element === 'string') {
               element = document.getElementById(element);
           }
           if (element) {
               element.innerHTML = `
                   <div class="message-success">
                       <i class="fas fa-check-circle mr-2"></i>
                       ${message}
                   </div>
               `;
           }
       },
       
       showNotification(message, type = 'info', duration = 5000) {
           const notification = document.createElement('div');
           notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
               type === 'success' ? 'bg-green-500' :
               type === 'error' ? 'bg-red-500' :
               type === 'warning' ? 'bg-yellow-500' :
               'bg-blue-500'
           } text-white`;
           
           notification.innerHTML = `
               <div class="flex items-center">
                   <span>${message}</span>
                   <button onclick="this.parentElement.parentElement.remove()" 
                           class="ml-3 text-white hover:text-gray-200">
                       <i class="fas fa-times"></i>
                   </button>
               </div>
           `;
           
           document.body.appendChild(notification);
           
           // Auto remove after duration
           if (duration > 0) {
               setTimeout(() => {
                   if (notification.parentElement) {
                       notification.remove();
                   }
               }, duration);
           }
       },
       
       formatBytes(bytes) {
           if (bytes === 0) return '0 Bytes';
           const k = 1024;
           const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
           const i = Math.floor(Math.log(bytes) / Math.log(k));
           return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
       },
       
       formatDuration(ms) {
           if (ms < 1000) return `${ms}ms`;
           if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
           if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
           return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
       },
       
       escapeHtml(text) {
           const map = {
               '&': '&amp;',
               '<': '&lt;',
               '>': '&gt;',
               '"': '&quot;',
               "'": '&#039;'
           };
           return text.replace(/[&<>"']/g, function(m) { return map[m]; });
       }
   },
   
   // Search utilities
   search: {
       highlightText(text, query) {
           if (!query) return text;
           
           const words = query.toLowerCase().split(/\s+/);
           let highlightedText = text;
           
           words.forEach(word => {
               if (word.length > 2) { // Only highlight words longer than 2 characters
                   const regex = new RegExp(`(${word})`, 'gi');
                   highlightedText = highlightedText.replace(regex, '<span class="search-highlight">$1</span>');
               }
           });
           
           return highlightedText;
       },
       
       truncateContent(content, maxLength = 300) {
           if (content.length <= maxLength) return content;
           
           // Try to find a sentence boundary near the limit
           const truncated = content.substring(0, maxLength);
           const lastSentence = truncated.lastIndexOf('.');
           const lastSpace = truncated.lastIndexOf(' ');
           
           const cutPoint = lastSentence > maxLength - 50 ? lastSentence + 1 : 
                          lastSpace > maxLength - 20 ? lastSpace : maxLength;
           
           return content.substring(0, cutPoint) + '...';
       }
   },
   
   // Storage utilities
   storage: {
       set(key, value) {
           try {
               localStorage.setItem(`rdb_${key}`, JSON.stringify(value));
           } catch (e) {
               console.warn('Could not save to localStorage:', e);
           }
       },
       
       get(key, defaultValue = null) {
           try {
               const item = localStorage.getItem(`rdb_${key}`);
               return item ? JSON.parse(item) : defaultValue;
           } catch (e) {
               console.warn('Could not read from localStorage:', e);
               return defaultValue;
           }
       },
       
       remove(key) {
           try {
               localStorage.removeItem(`rdb_${key}`);
           } catch (e) {
               console.warn('Could not remove from localStorage:', e);
           }
       }
   }
};

// Global event handlers
document.addEventListener('DOMContentLoaded', function() {
   // Initialize tooltips if any exist
   initializeTooltips();
   
   // Initialize keyboard shortcuts
   initializeKeyboardShortcuts();
   
   // Set up global error handler
   setupGlobalErrorHandler();
   
   // Load user preferences
   loadUserPreferences();
});

function initializeTooltips() {
   // Simple tooltip implementation
   const tooltipElements = document.querySelectorAll('[data-tooltip]');
   
   tooltipElements.forEach(element => {
       element.addEventListener('mouseenter', function() {
           const tooltip = document.createElement('div');
           tooltip.className = 'absolute bg-gray-800 text-white px-2 py-1 rounded text-sm z-50';
           tooltip.textContent = this.getAttribute('data-tooltip');
           tooltip.id = 'tooltip';
           
           document.body.appendChild(tooltip);
           
           const rect = this.getBoundingClientRect();
           tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
           tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
       });
       
       element.addEventListener('mouseleave', function() {
           const tooltip = document.getElementById('tooltip');
           if (tooltip) {
               tooltip.remove();
           }
       });
   });
}

function initializeKeyboardShortcuts() {
   document.addEventListener('keydown', function(e) {
       // Ctrl/Cmd + / for search focus
       if ((e.ctrlKey || e.metaKey) && e.key === '/') {
           e.preventDefault();
           const searchInput = document.getElementById('searchQuery') || document.getElementById('quickSearch');
           if (searchInput) {
               searchInput.focus();
           }
       }
       
       // Escape to close modals
       if (e.key === 'Escape') {
           const modals = document.querySelectorAll('.modal-backdrop, [id$="Modal"]');
           modals.forEach(modal => {
               if (!modal.classList.contains('hidden')) {
                   modal.classList.add('hidden');
               }
           });
       }
   });
}

function setupGlobalErrorHandler() {
   window.addEventListener('error', function(e) {
       console.error('Global error:', e.error);
       RDB.ui.showNotification('An unexpected error occurred', 'error');
   });
   
   window.addEventListener('unhandledrejection', function(e) {
       console.error('Unhandled promise rejection:', e.reason);
       RDB.ui.showNotification('Network error occurred', 'error');
   });
}

function loadUserPreferences() {
   // Load and apply saved preferences
   const preferences = RDB.storage.get('preferences', {
       theme: 'light',
       searchResultsPerPage: 10,
       autoRefresh: true
   });
   
   // Apply theme
   if (preferences.theme === 'dark') {
       document.body.classList.add('dark');
   }
   
   // Store preferences globally
   window.userPreferences = preferences;
}

function saveUserPreferences() {
   if (window.userPreferences) {
       RDB.storage.set('preferences', window.userPreferences);
   }
}

// Utility functions for common operations
function debounce(func, wait) {
   let timeout;
   return function executedFunction(...args) {
       const later = () => {
           clearTimeout(timeout);
           func(...args);
       };
       clearTimeout(timeout);
       timeout = setTimeout(later, wait);
   };
}

function throttle(func, limit) {
   let inThrottle;
   return function() {
       const args = arguments;
       const context = this;
       if (!inThrottle) {
           func.apply(context, args);
           inThrottle = true;
           setTimeout(() => inThrottle = false, limit);
       }
   };
}

// Auto-save form data
function setupAutoSave(formId, storageKey) {
   const form = document.getElementById(formId);
   if (!form) return;
   
   // Load saved data
   const savedData = RDB.storage.get(storageKey, {});
   
   // Restore form values
   Object.keys(savedData).forEach(key => {
       const input = form.querySelector(`[name="${key}"]`);
       if (input) {
           if (input.type === 'checkbox') {
               input.checked = savedData[key];
           } else {
               input.value = savedData[key];
           }
       }
   });
   
   // Save on change
   const debouncedSave = debounce(() => {
       const formData = new FormData(form);
       const data = {};
       
       for (let [key, value] of formData.entries()) {
           const input = form.querySelector(`[name="${key}"]`);
           if (input && input.type === 'checkbox') {
               data[key] = input.checked;
           } else {
               data[key] = value;
           }
       }
       
       RDB.storage.set(storageKey, data);
   }, 500);
   
   form.addEventListener('input', debouncedSave);
   form.addEventListener('change', debouncedSave);
}

// Copy to clipboard functionality
function copyToClipboard(text) {
   if (navigator.clipboard) {
       navigator.clipboard.writeText(text).then(() => {
           RDB.ui.showNotification('Copied to clipboard', 'success', 2000);
       }).catch(() => {
           fallbackCopyToClipboard(text);
       });
   } else {
       fallbackCopyToClipboard(text);
   }
}

function fallbackCopyToClipboard(text) {
   const textArea = document.createElement('textarea');
   textArea.value = text;
   textArea.style.position = 'fixed';
   textArea.style.left = '-999999px';
   textArea.style.top = '-999999px';
   document.body.appendChild(textArea);
   textArea.focus();
   textArea.select();
   
   try {
       document.execCommand('copy');
       RDB.ui.showNotification('Copied to clipboard', 'success', 2000);
   } catch (err) {
       RDB.ui.showNotification('Could not copy to clipboard', 'error');
   }
   
   document.body.removeChild(textArea);
}

// Export global functions
window.copyToClipboard = copyToClipboard;
window.debounce = debounce;
window.throttle = throttle;
window.setupAutoSave = setupAutoSave;
