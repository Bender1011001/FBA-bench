import { notificationService, NotificationService } from '../notificationService';

// Mock document methods
const mockCreateElement = jest.fn();
const mockAppendChild = jest.fn();
const mockRemoveChild = jest.fn();
const mockSetTimeout = jest.fn();
const mockClearTimeout = jest.fn();

// Mock DOM elements
const mockElement = {
  className: '',
  innerHTML: '',
  addEventListener: jest.fn(),
  classList: {
    add: jest.fn(),
    remove: jest.fn(),
    contains: jest.fn()
  },
  remove: jest.fn()
};

const mockContainer = {
  appendChild: mockAppendChild,
  innerHTML: '',
  style: {},
  removeChild: mockRemoveChild
};

describe('NotificationService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup DOM mocks
    document.createElement = mockCreateElement.mockReturnValue(mockElement);
    document.body.appendChild = jest.fn();
    document.body.removeChild = jest.fn();
    
    // Setup timer mocks
    global.setTimeout = mockSetTimeout as any;
    global.clearTimeout = mockClearTimeout as any;
    
    // Create a new instance for each test
    notificationService = new NotificationService();
  });

  describe('show', () => {
    test('creates a notification with default options', () => {
      notificationService.show('Test message');
      
      expect(mockCreateElement).toHaveBeenCalledWith('div');
      expect(mockAppendChild).toHaveBeenCalledWith(mockElement);
    });

    test('creates a notification with custom options', () => {
      const customAction = {
        label: 'Test Action',
        callback: jest.fn()
      };
      
      notificationService.show('Test message', {
        type: 'success',
        duration: 10000,
        position: 'top-left',
        action: customAction
      });
      
      expect(mockCreateElement).toHaveBeenCalledWith('div');
      expect(mockAppendChild).toHaveBeenCalledWith(mockElement);
    });

    test('adds event listeners to close button', () => {
      notificationService.show('Test message');
      
      expect(mockElement.addEventListener).toHaveBeenCalledWith(
        'click',
        expect.any(Function)
      );
    });

    test('adds event listeners to action button if provided', () => {
      const customAction = {
        label: 'Test Action',
        callback: jest.fn()
      };
      
      notificationService.show('Test message', {
        action: customAction
      });
      
      // Should have two event listeners (close button and action button)
      expect(mockElement.addEventListener).toHaveBeenCalledTimes(2);
    });

    test('sets auto-remove timeout if duration > 0', () => {
      notificationService.show('Test message', { duration: 5000 });
      
      expect(mockSetTimeout).toHaveBeenCalledWith(
        expect.any(Function),
        5000
      );
    });

    test('does not set auto-remove timeout if duration is 0', () => {
      notificationService.show('Test message', { duration: 0 });
      
      expect(mockSetTimeout).not.toHaveBeenCalled();
    });
  });

  describe('info', () => {
    test('shows info notification', () => {
      notificationService.info('Info message');
      
      expect(mockCreateElement).toHaveBeenCalledWith('div');
    });

    test('accepts custom duration', () => {
      notificationService.info('Info message', 10000);
      
      expect(mockSetTimeout).toHaveBeenCalledWith(
        expect.any(Function),
        10000
      );
    });
  });

  describe('success', () => {
    test('shows success notification', () => {
      notificationService.success('Success message');
      
      expect(mockCreateElement).toHaveBeenCalledWith('div');
    });

    test('accepts custom duration', () => {
      notificationService.success('Success message', 10000);
      
      expect(mockSetTimeout).toHaveBeenCalledWith(
        expect.any(Function),
        10000
      );
    });
  });

  describe('warning', () => {
    test('shows warning notification', () => {
      notificationService.warning('Warning message');
      
      expect(mockCreateElement).toHaveBeenCalledWith('div');
    });

    test('accepts custom duration', () => {
      notificationService.warning('Warning message', 10000);
      
      expect(mockSetTimeout).toHaveBeenCalledWith(
        expect.any(Function),
        10000
      );
    });
  });

  describe('error', () => {
    test('shows error notification', () => {
      notificationService.error('Error message');
      
      expect(mockCreateElement).toHaveBeenCalledWith('div');
    });

    test('accepts custom duration', () => {
      notificationService.error('Error message', 10000);
      
      expect(mockSetTimeout).toHaveBeenCalledWith(
        expect.any(Function),
        10000
      );
    });
  });

  describe('clear', () => {
    test('clears all notifications', () => {
      // Add some notifications
      notificationService.info('Test 1');
      notificationService.success('Test 2');
      
      // Clear all
      notificationService.clear();
      
      expect(mockContainer.innerHTML).toBe('');
    });
  });

  describe('getNotifications', () => {
    test('returns a copy of notifications', () => {
      notificationService.info('Test 1');
      notificationService.success('Test 2');
      
      const notifications = notificationService.getNotifications();
      
      expect(notifications).toHaveLength(2);
      expect(notifications[0].message).toBe('Test 1');
      expect(notifications[1].message).toBe('Test 2');
    });
  });

  describe('subscribe', () => {
    test('adds listener and returns unsubscribe function', () => {
      const listener = jest.fn();
      const unsubscribe = notificationService.subscribe(listener);
      
      // Trigger a notification
      notificationService.info('Test message');
      
      expect(listener).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            message: 'Test message',
            type: 'info'
          })
        ])
      );
      
      // Unsubscribe
      unsubscribe();
      
      // Trigger another notification
      notificationService.success('Another message');
      
      // Listener should not be called again
      expect(listener).toHaveBeenCalledTimes(1);
    });
  });

  describe('getIcon', () => {
    test('returns correct icon for info type', () => {
      const service = new NotificationService();
      // @ts-ignore - Accessing private method for testing
      const icon = service.getIcon('info');
      
      expect(icon).toContain('text-blue-400');
    });

    test('returns correct icon for success type', () => {
      const service = new NotificationService();
      // @ts-ignore - Accessing private method for testing
      const icon = service.getIcon('success');
      
      expect(icon).toContain('text-green-400');
    });

    test('returns correct icon for warning type', () => {
      const service = new NotificationService();
      // @ts-ignore - Accessing private method for testing
      const icon = service.getIcon('warning');
      
      expect(icon).toContain('text-yellow-400');
    });

    test('returns correct icon for error type', () => {
      const service = new NotificationService();
      // @ts-ignore - Accessing private method for testing
      const icon = service.getIcon('error');
      
      expect(icon).toContain('text-red-400');
    });
  });

  describe('removeNotification', () => {
    test('removes notification by ID', () => {
      // Add a notification
      notificationService.info('Test message');
      
      // Get the notification ID
      const notifications = notificationService.getNotifications();
      const id = notifications[0].id;
      
      // Remove the notification
      // @ts-ignore - Accessing private method for testing
      notificationService.removeNotification(id);
      
      // Check that the notification was removed
      const updatedNotifications = notificationService.getNotifications();
      expect(updatedNotifications).toHaveLength(0);
    });

    test('does nothing if notification ID not found', () => {
      // Add a notification
      notificationService.info('Test message');
      
      // Try to remove a non-existent notification
      // @ts-ignore - Accessing private method for testing
      notificationService.removeNotification('non-existent-id');
      
      // Check that the notification was not removed
      const notifications = notificationService.getNotifications();
      expect(notifications).toHaveLength(1);
    });
  });

  describe('notifyListeners', () => {
    test('calls all listeners with current notifications', () => {
      const listener1 = jest.fn();
      const listener2 = jest.fn();
      
      notificationService.subscribe(listener1);
      notificationService.subscribe(listener2);
      
      // Add a notification
      notificationService.info('Test message');
      
      expect(listener1).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            message: 'Test message',
            type: 'info'
          })
        ])
      );
      
      expect(listener2).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            message: 'Test message',
            type: 'info'
          })
        ])
      );
    });
  });
});