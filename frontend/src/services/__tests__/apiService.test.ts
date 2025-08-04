import { apiService } from '../apiService';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('apiService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('get', () => {
    test('makes a GET request with correct URL and config', async () => {
      const mockResponse = { data: { test: 'data' } };
      mockedAxios.get.mockResolvedValue(mockResponse);

      const result = await apiService.get('/test-endpoint', {
        headers: { 'Custom-Header': 'value' }
      });

      expect(mockedAxios.get).toHaveBeenCalledWith(
        '/test-endpoint',
        {
          headers: { 'Custom-Header': 'value' }
        }
      );
      expect(result).toEqual(mockResponse);
    });

    test('handles errors correctly', async () => {
      const mockError = new Error('Network error');
      mockedAxios.get.mockRejectedValue(mockError);

      await expect(apiService.get('/test-endpoint')).rejects.toThrow('Network error');
    });
  });

  describe('post', () => {
    test('makes a POST request with correct URL, data, and config', async () => {
      const mockData = { name: 'test' };
      const mockResponse = { data: { success: true } };
      mockedAxios.post.mockResolvedValue(mockResponse);

      const result = await apiService.post('/test-endpoint', mockData, {
        headers: { 'Content-Type': 'application/json' }
      });

      expect(mockedAxios.post).toHaveBeenCalledWith(
        '/test-endpoint',
        mockData,
        {
          headers: { 'Content-Type': 'application/json' }
        }
      );
      expect(result).toEqual(mockResponse);
    });

    test('handles errors correctly', async () => {
      const mockError = new Error('Network error');
      mockedAxios.post.mockRejectedValue(mockError);

      await expect(apiService.post('/test-endpoint', {})).rejects.toThrow('Network error');
    });
  });

  describe('put', () => {
    test('makes a PUT request with correct URL, data, and config', async () => {
      const mockData = { name: 'test' };
      const mockResponse = { data: { success: true } };
      mockedAxios.put.mockResolvedValue(mockResponse);

      const result = await apiService.put('/test-endpoint', mockData, {
        headers: { 'Content-Type': 'application/json' }
      });

      expect(mockedAxios.put).toHaveBeenCalledWith(
        '/test-endpoint',
        mockData,
        {
          headers: { 'Content-Type': 'application/json' }
        }
      );
      expect(result).toEqual(mockResponse);
    });

    test('handles errors correctly', async () => {
      const mockError = new Error('Network error');
      mockedAxios.put.mockRejectedValue(mockError);

      await expect(apiService.put('/test-endpoint', {})).rejects.toThrow('Network error');
    });
  });

  describe('delete', () => {
    test('makes a DELETE request with correct URL and config', async () => {
      const mockResponse = { data: { success: true } };
      mockedAxios.delete.mockResolvedValue(mockResponse);

      const result = await apiService.delete('/test-endpoint', {
        headers: { 'Custom-Header': 'value' }
      });

      expect(mockedAxios.delete).toHaveBeenCalledWith(
        '/test-endpoint',
        {
          headers: { 'Custom-Header': 'value' }
        }
      );
      expect(result).toEqual(mockResponse);
    });

    test('handles errors correctly', async () => {
      const mockError = new Error('Network error');
      mockedAxios.delete.mockRejectedValue(mockError);

      await expect(apiService.delete('/test-endpoint')).rejects.toThrow('Network error');
    });
  });

  describe('patch', () => {
    test('makes a PATCH request with correct URL, data, and config', async () => {
      const mockData = { name: 'test' };
      const mockResponse = { data: { success: true } };
      mockedAxios.patch.mockResolvedValue(mockResponse);

      const result = await apiService.patch('/test-endpoint', mockData, {
        headers: { 'Content-Type': 'application/json' }
      });

      expect(mockedAxios.patch).toHaveBeenCalledWith(
        '/test-endpoint',
        mockData,
        {
          headers: { 'Content-Type': 'application/json' }
        }
      );
      expect(result).toEqual(mockResponse);
    });

    test('handles errors correctly', async () => {
      const mockError = new Error('Network error');
      mockedAxios.patch.mockRejectedValue(mockError);

      await expect(apiService.patch('/test-endpoint', {})).rejects.toThrow('Network error');
    });
  });

  describe('setAuthToken', () => {
    test('sets the Authorization header with Bearer token', () => {
      apiService.setAuthToken('test-token');

      expect(mockedAxios.defaults.headers.common['Authorization']).toBe('Bearer test-token');
    });

    test('removes the Authorization header when token is null', () => {
      apiService.setAuthToken('test-token');
      expect(mockedAxios.defaults.headers.common['Authorization']).toBe('Bearer test-token');

      apiService.setAuthToken(null);
      expect(mockedAxios.defaults.headers.common['Authorization']).toBeUndefined();
    });
  });

  describe('setBaseURL', () => {
    test('sets the base URL for all requests', () => {
      apiService.setBaseURL('https://api.example.com');

      expect(mockedAxios.defaults.baseURL).toBe('https://api.example.com');
    });
  });

  describe('setTimeout', () => {
    test('sets the timeout for all requests', () => {
      apiService.setTimeout(10000);

      expect(mockedAxios.defaults.timeout).toBe(10000);
    });
  });

  describe('setHeaders', () => {
    test('sets custom headers for all requests', () => {
      const headers = {
        'Content-Type': 'application/json',
        'X-Custom-Header': 'custom-value'
      };

      apiService.setHeaders(headers);

      expect(mockedAxios.defaults.headers.common['Content-Type']).toBe('application/json');
      expect(mockedAxios.defaults.headers.common['X-Custom-Header']).toBe('custom-value');
    });
  });

  describe('interceptors', () => {
    test('adds request interceptor', () => {
      const interceptor = (config: any) => {
        config.headers['X-Test'] = 'test-value';
        return config;
      };

      const interceptorId = apiService.addRequestInterceptor(interceptor);
      
      expect(typeof interceptorId).toBe('number');
      expect(mockedAxios.interceptors.request.use).toHaveBeenCalledWith(interceptor);
    });

    test('adds response interceptor', () => {
      const successInterceptor = (response: any) => response;
      const errorInterceptor = (error: any) => Promise.reject(error);

      const interceptorId = apiService.addResponseInterceptor(successInterceptor, errorInterceptor);
      
      expect(typeof interceptorId).toBe('number');
      expect(mockedAxios.interceptors.response.use).toHaveBeenCalledWith(successInterceptor, errorInterceptor);
    });

    test('removes request interceptor', () => {
      const interceptorId = 1;
      
      apiService.removeRequestInterceptor(interceptorId);
      
      expect(mockedAxios.interceptors.request.eject).toHaveBeenCalledWith(interceptorId);
    });

    test('removes response interceptor', () => {
      const interceptorId = 1;
      
      apiService.removeResponseInterceptor(interceptorId);
      
      expect(mockedAxios.interceptors.response.eject).toHaveBeenCalledWith(interceptorId);
    });
  });

  describe('create', () => {
    test('creates a new axios instance with custom config', () => {
      const config = {
        baseURL: 'https://api.example.com',
        timeout: 5000,
        headers: {
          'X-Custom-Header': 'custom-value'
        }
      };

      const newApiService = apiService.create(config);

      expect(mockedAxios.create).toHaveBeenCalledWith(config);
    });
  });
});