// frontend/src/utils/notificationService.ts

import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid'; // will need to install uuid
import { enableMapSet } from 'immer'; // will need to install immer to use enableMapSet
import { handleError, type AppError, ErrorCategory } from './errorHandler'; // Use existing error system

// Enable Map and Set Immer compatibility
enableMapSet();

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  message: string;
  type: NotificationType;
  duration?: number; // Milliseconds, 0 for persistent
  action?: {
    label: string;
    onClick: () => void;
  };
  details?: string; // Optional detailed information
  timestamp: Date;
}

interface NotificationState {
  notifications: Map<string, Notification>;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  removeNotification: (id: string) => void;
  clearAllNotifications: () => void;
  // Handler for AppErrors, converting them into notifications
  showErrorNotification: (error: AppError) => void;
}

export const useNotificationStore = create<NotificationState>((set, get) => ({
  notifications: new Map<string, Notification>(),

  addNotification: (notification) => {
    const id = uuidv4();
    const newNotification: Notification = {
      ...notification,
      id,
      timestamp: new Date(),
    };
    set((state) => {
      const newNotifications = new Map(state.notifications);
      newNotifications.set(id, newNotification);
      return { notifications: newNotifications };
    });

    if (notification.duration && notification.duration > 0) {
      setTimeout(() => {
        get().removeNotification(id);
      }, notification.duration);
    }
    return id;
  },

  removeNotification: (id) => {
    set((state) => {
      const newNotifications = new Map(state.notifications);
      newNotifications.delete(id);
      return { notifications: newNotifications };
    });
  },

  clearAllNotifications: () => {
    set({ notifications: new Map() });
  },

  showErrorNotification: (error: AppError) => {
    const defaultErrorMessage = 'An unknown error occurred.';
    get().addNotification({
      message: error.userMessage || error.message || defaultErrorMessage,
      type: 'error',
      duration: error.category === ErrorCategory.System || error.category === ErrorCategory.Network ? 0 : 5000,
      details: error.details ? JSON.stringify(error.details, null, 2) : error.stack,
      action: {
        label: 'Details',
        onClick: () => {
          // In a real app, this might open a modal with more info
          console.log('Error details:', error);
          alert('Check console for error details. A dedicated error details modal would show here.');
        },
      }
    });
    handleError(error); // Keep logging error via the centralized handler
  },
}));

// Export a simpler interface for direct usage
export const notificationService = {
  // Directly add a success notification
  success: (message: string, duration: number = 3000) => {
    return useNotificationStore.getState().addNotification({ message, type: 'success', duration });
  },
  // Directly add an info notification
  info: (message: string, duration: number = 4000) => {
    return useNotificationStore.getState().addNotification({ message, type: 'info', duration });
  },
  // Directly add a warning notification
  warning: (message: string, duration: number = 5000) => {
    return useNotificationStore.getState().addNotification({ message, type: 'warning', duration });
  },
  // Directly add an error notification (persistent by default for general errors)
  error: (message: string, duration: number = 0, details?: string, action?: { label: string; onClick: () => void; }) => {
    return useNotificationStore.getState().addNotification({ message, type: 'error', duration, details, action });
  },
  // Show a notification from an AppError
  showAppError: (error: AppError) => useNotificationStore.getState().showErrorNotification(error),
  // Clear specific or all notifications
  remove: (id: string) => useNotificationStore.getState().removeNotification(id),
  clearAll: () => useNotificationStore.getState().clearAllNotifications(),
};