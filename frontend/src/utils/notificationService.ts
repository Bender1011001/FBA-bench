interface NotificationOptions {
  duration?: number;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  type?: 'info' | 'success' | 'warning' | 'error';
  action?: {
    label: string;
    callback: () => void;
  };
}

interface Notification {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  duration: number;
  position: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  action?: {
    label: string;
    callback: () => void;
  };
  timestamp: number;
}

class NotificationService {
  private notifications: Notification[] = [];
  private listeners: ((notifications: Notification[]) => void)[] = [];
  private container: HTMLDivElement | null = null;

  constructor() {
    this.initializeContainer();
  }

  private initializeContainer() {
    if (typeof document !== 'undefined') {
      // Create container for notifications
      this.container = document.createElement('div');
      this.container.className = 'fixed z-50 flex flex-col gap-2 p-4 pointer-events-none';
      this.container.style.top = '20px';
      this.container.style.right = '20px';
      document.body.appendChild(this.container);
    }
  }

  private createNotificationElement(notification: Notification): HTMLDivElement {
    const element = document.createElement('div');
    element.className = `notification-item transform transition-all duration-300 translate-x-full opacity-0 max-w-sm rounded-lg shadow-lg pointer-events-auto`;
    
    // Set background color based on type
    switch (notification.type) {
      case 'success':
        element.classList.add('bg-green-50', 'border', 'border-green-200');
        break;
      case 'error':
        element.classList.add('bg-red-50', 'border', 'border-red-200');
        break;
      case 'warning':
        element.classList.add('bg-yellow-50', 'border', 'border-yellow-200');
        break;
      case 'info':
      default:
        element.classList.add('bg-blue-50', 'border', 'border-blue-200');
        break;
    }

    element.innerHTML = `
      <div class="flex items-start p-4">
        <div class="flex-shrink-0">
          ${this.getIcon(notification.type)}
        </div>
        <div class="ml-3 w-0 flex-1">
          <p class="text-sm font-medium text-gray-900">${notification.message}</p>
        </div>
        <div class="ml-4 flex-shrink-0 flex">
          <button class="notification-close inline-flex text-gray-400 hover:text-gray-500 focus:outline-none">
            <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
      ${notification.action ? `
        <div class="bg-gray-50 px-4 py-3 sm:px-6">
          <div class="flex">
            <button class="notification-action w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none sm:ml-3 sm:w-auto sm:text-sm">
              ${notification.action.label}
            </button>
          </div>
        </div>
      ` : ''}
    `;

    // Add event listeners
    const closeButton = element.querySelector('.notification-close');
    if (closeButton) {
      closeButton.addEventListener('click', () => {
        this.removeNotification(notification.id);
      });
    }

    const actionButton = element.querySelector('.notification-action');
    if (actionButton && notification.action) {
      actionButton.addEventListener('click', () => {
        notification.action?.callback();
        this.removeNotification(notification.id);
      });
    }

    return element;
  }

  private getIcon(type: string): string {
    switch (type) {
      case 'success':
        return `
          <svg class="h-6 w-6 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        `;
      case 'error':
        return `
          <svg class="h-6 w-6 text-red-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        `;
      case 'warning':
        return `
          <svg class="h-6 w-6 text-yellow-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        `;
      case 'info':
      default:
        return `
          <svg class="h-6 w-6 text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        `;
    }
  }

  private addNotification(notification: Notification) {
    this.notifications.push(notification);
    this.notifyListeners();
    
    if (this.container) {
      const element = this.createNotificationElement(notification);
      this.container.appendChild(element);
      
      // Trigger animation
      setTimeout(() => {
        element.classList.remove('translate-x-full', 'opacity-0');
      }, 10);
      
      // Auto-remove after duration
      if (notification.duration > 0) {
        setTimeout(() => {
          this.removeNotification(notification.id);
        }, notification.duration);
      }
    }
  }

  private removeNotification(id: string) {
    const index = this.notifications.findIndex(n => n.id === id);
    if (index !== -1) {
      this.notifications.splice(index, 1);
      this.notifyListeners();
      
      if (this.container) {
        const element = this.container.querySelector(`[data-notification-id="${id}"]`);
        if (element) {
          element.classList.add('translate-x-full', 'opacity-0');
          setTimeout(() => {
            element.remove();
          }, 300);
        }
      }
    }
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener([...this.notifications]));
  }

  public subscribe(listener: (notifications: Notification[]) => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  public show(message: string, options: NotificationOptions = {}) {
    const notification: Notification = {
      id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      message,
      type: options.type || 'info',
      duration: options.duration || 5000,
      position: options.position || 'top-right',
      action: options.action,
      timestamp: Date.now()
    };

    this.addNotification(notification);
  }

  public info(message: string, duration?: number) {
    this.show(message, { type: 'info', duration });
  }

  public success(message: string, duration?: number) {
    this.show(message, { type: 'success', duration });
  }

  public warning(message: string, duration?: number) {
    this.show(message, { type: 'warning', duration });
  }

  public error(message: string, duration?: number) {
    this.show(message, { type: 'error', duration });
  }

  public clear() {
    this.notifications = [];
    this.notifyListeners();
    
    if (this.container) {
      this.container.innerHTML = '';
    }
  }

  public getNotifications(): Notification[] {
    return [...this.notifications];
  }
}

import { useState, useEffect } from 'react';

// Create singleton instance
export const notificationService = new NotificationService();

// Export class for testing or multiple instances
export { NotificationService };

// React hook for using the notification service
export function useNotificationStore() {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  useEffect(() => {
    // Subscribe to notification updates
    const unsubscribe = notificationService.subscribe((updatedNotifications) => {
      setNotifications(updatedNotifications);
    });

    // Get initial notifications
    setNotifications(notificationService.getNotifications());

    return unsubscribe;
  }, []);

  const removeNotification = (id: string) => {
    // Since the service handles its own DOM, we just need to trigger removal
    const currentNotifications = notificationService.getNotifications();
    const index = currentNotifications.findIndex(n => n.id === id);
    if (index !== -1) {
      // Remove from internal array and notify listeners
      currentNotifications.splice(index, 1);
      notificationService.clear();
      currentNotifications.forEach(n => notificationService.show(n.message, {
        type: n.type,
        duration: n.duration,
        position: n.position,
        action: n.action
      }));
    }
  };

  return {
    notifications: new Map(notifications.map(n => [n.id, n])),
    removeNotification
  };
}

// Export types
export type { Notification, NotificationOptions };