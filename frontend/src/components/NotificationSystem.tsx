// frontend/src/components/NotificationSystem.tsx

import React from 'react';
import { useNotificationStore } from '../utils/notificationService'; // Assuming notificationService.ts is in utils

const NotificationSystem: React.FC = () => {
  const { notifications, removeNotification } = useNotificationStore();

  if (notifications.size === 0) {
    return null;
  }

  // Convert Map to Array for rendering, ensuring timestamps are parsed as Dates for sorting
  const notificationsArray = Array.from(notifications.values()).sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col items-end space-y-3">
      {notificationsArray.map((notification) => (
        <div
          key={notification.id}
          className={`flex items-center justify-between p-4 rounded-lg shadow-lg text-white max-w-sm w-full transition-all duration-300 ease-out transform
            ${notification.type === 'success' ? 'bg-green-500' : ''}
            ${notification.type === 'error' ? 'bg-red-600' : ''}
            ${notification.type === 'warning' ? 'bg-yellow-500' : ''}
            ${notification.type === 'info' ? 'bg-blue-500' : ''}
          `}
          role="alert"
        >
          <div className="flex flex-col flex-grow">
            <p className="font-semibold">{notification.message}</p>
            {notification.action && (
              <button
                className="mt-2 px-3 py-1 bg-white bg-opacity-20 hover:bg-opacity-30 rounded text-sm self-start"
                onClick={notification.action.callback}
              >
                {notification.action.label}
              </button>
            )}
          </div>
          <button
            onClick={() => removeNotification(notification.id)}
            className="ml-4 p-1 rounded-full hover:bg-white hover:bg-opacity-20 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-50"
            aria-label="Close notification"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  );
};

export default NotificationSystem;