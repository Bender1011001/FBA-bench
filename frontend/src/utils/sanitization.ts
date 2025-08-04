// frontend/src/utils/sanitization.ts

import DOMPurify from 'dompurify';

/**
 * Configuration options for DOMPurify sanitization
 */
interface SanitizationOptions {
  ALLOWED_TAGS?: string[];
  ALLOWED_ATTR?: string[];
  ALLOW_DATA_ATTR?: boolean;
  FORCE_BODY?: boolean;
  SANITIZE_DOM?: boolean;
  ALLOW_UNKNOWN_PROTOCOLS?: boolean;
  USE_PROFILES?: {
    html?: boolean;
    svg?: boolean;
    svgFilters?: boolean;
    mathMl?: boolean;
    text?: boolean;
  };
}

/**
 * Default sanitization options for text content (no HTML allowed)
 */
const DEFAULT_TEXT_OPTIONS: SanitizationOptions = {
  ALLOWED_TAGS: [],
  ALLOWED_ATTR: [],
  ALLOW_DATA_ATTR: false,
  FORCE_BODY: false,
  SANITIZE_DOM: true,
  ALLOW_UNKNOWN_PROTOCOLS: false,
};

/**
 * Default sanitization options for safe HTML content
 */
const DEFAULT_HTML_OPTIONS: SanitizationOptions = {
  ALLOWED_TAGS: [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'br', 'hr',
    'ul', 'ol', 'li',
    'strong', 'em', 'i', 'b', 'u',
    'a', 'img',
    'blockquote', 'code', 'pre',
    'div', 'span',
    'table', 'thead', 'tbody', 'tr', 'th', 'td'
  ],
  ALLOWED_ATTR: [
    'href', 'src', 'alt', 'title', 'class', 'id'
  ],
  ALLOW_DATA_ATTR: false,
  FORCE_BODY: false,
  SANITIZE_DOM: true,
  ALLOW_UNKNOWN_PROTOCOLS: false,
};

/**
 * Sanitizes text content by removing all HTML tags and potentially dangerous content
 * @param text The text to sanitize
 * @param options Optional sanitization options
 * @returns Sanitized text
 */
export const sanitizeText = (text: string, options: SanitizationOptions = DEFAULT_TEXT_OPTIONS): string => {
  try {
    if (typeof text !== 'string') {
      console.warn('sanitizeText: Input is not a string, converting to string');
      text = String(text);
    }
    
    return DOMPurify.sanitize(text, options);
  } catch (error) {
    console.error('Error sanitizing text:', error);
    return '[Sanitization Error]';
  }
};

/**
 * Sanitizes HTML content, allowing only safe HTML tags and attributes
 * @param html The HTML content to sanitize
 * @param options Optional sanitization options
 * @returns Sanitized HTML
 */
export const sanitizeHTML = (html: string, options: SanitizationOptions = DEFAULT_HTML_OPTIONS): string => {
  try {
    if (typeof html !== 'string') {
      console.warn('sanitizeHTML: Input is not a string, converting to string');
      html = String(html);
    }
    
    return DOMPurify.sanitize(html, options);
  } catch (error) {
    console.error('Error sanitizing HTML:', error);
    return '[Sanitization Error]';
  }
};

/**
 * Sanitizes a URL to prevent XSS attacks
 * @param url The URL to sanitize
 * @returns Sanitized URL or empty string if invalid
 */
export const sanitizeURL = (url: string): string => {
  try {
    if (typeof url !== 'string') {
      return '';
    }
    
    // Remove potentially dangerous protocols
    const sanitizedUrl = url.replace(/^(javascript|data|vbscript):/i, '');
    
    // Basic URL validation
    if (!sanitizedUrl || sanitizedUrl.trim() === '') {
      return '';
    }
    
    return sanitizedUrl;
  } catch (error) {
    console.error('Error sanitizing URL:', error);
    return '';
  }
};

/**
 * Sanitizes a numeric value to ensure it's a safe number
 * @param value The value to sanitize
 * @param defaultValue Default value if sanitization fails
 * @returns Sanitized number
 */
export const sanitizeNumber = (value: unknown, defaultValue: number = 0): number => {
  try {
    const num = Number(value);
    
    if (isNaN(num) || !isFinite(num)) {
      return defaultValue;
    }
    
    return num;
  } catch (error) {
    console.error('Error sanitizing number:', error);
    return defaultValue;
  }
};

/**
 * Sanitizes an integer value
 * @param value The value to sanitize
 * @param defaultValue Default value if sanitization fails
 * @returns Sanitized integer
 */
export const sanitizeInteger = (value: unknown, defaultValue: number = 0): number => {
  const num = sanitizeNumber(value, defaultValue);
  return Math.floor(num);
};

/**
 * Sanitizes a positive number
 * @param value The value to sanitize
 * @param defaultValue Default value if sanitization fails
 * @returns Sanitized positive number
 */
export const sanitizePositiveNumber = (value: unknown, defaultValue: number = 0): number => {
  const num = sanitizeNumber(value, defaultValue);
  return Math.max(0, num);
};

/**
 * Sanitizes a boolean value
 * @param value The value to sanitize
 * @param defaultValue Default value if sanitization fails
 * @returns Sanitized boolean
 */
export const sanitizeBoolean = (value: unknown, defaultValue: boolean = false): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  
  if (typeof value === 'string') {
    const lowerValue = value.toLowerCase();
    return lowerValue === 'true' || lowerValue === '1' || lowerValue === 'yes';
  }
  
  if (typeof value === 'number') {
    return value === 1;
  }
  
  return defaultValue;
};

/**
 * Sanitizes an array of strings
 * @param array The array to sanitize
 * @returns Sanitized array of strings
 */
export const sanitizeStringArray = (array: unknown[]): string[] => {
  try {
    if (!Array.isArray(array)) {
      return [];
    }
    
    return array
      .filter(item => typeof item === 'string')
      .map(item => sanitizeText(item));
  } catch (error) {
    console.error('Error sanitizing string array:', error);
    return [];
  }
};

/**
 * Sanitizes an object by recursively sanitizing all string properties
 * @param obj The object to sanitize
 * @returns Sanitized object
 */
export const sanitizeObject = <T extends Record<string, unknown>>(obj: T): T => {
  try {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }
    
    const result: Record<string, unknown> = {};
    
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'string') {
        result[key] = sanitizeText(value);
      } else if (Array.isArray(value)) {
        result[key] = value.map(item => {
          if (typeof item === 'string') {
            return sanitizeText(item);
          } else if (typeof item === 'object' && item !== null) {
            return sanitizeObject(item as Record<string, unknown>);
          }
          return item;
        });
      } else if (typeof value === 'object' && value !== null) {
        result[key] = sanitizeObject(value as Record<string, unknown>);
      } else {
        result[key] = value;
      }
    }
    
    return result as T;
  } catch (error) {
    console.error('Error sanitizing object:', error);
    return obj;
  }
};

/**
 * Validates and sanitizes a timestamp
 * @param timestamp The timestamp to validate and sanitize
 * @returns Sanitized timestamp or 'Invalid time' if invalid
 */
export const sanitizeTimestamp = (timestamp: string): string => {
  try {
    if (typeof timestamp !== 'string' || timestamp.length < 10) {
      return 'Invalid time';
    }
    
    // Attempt to create a date object
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      return 'Invalid time';
    }
    
    return date.toLocaleTimeString();
  } catch (error) {
    console.error('Error sanitizing timestamp:', error);
    return 'Invalid time';
  }
};

/**
 * Sanitizes user input from form fields
 * @param input The user input to sanitize
 * @returns Sanitized input
 */
export const sanitizeUserInput = (input: string): string => {
  return sanitizeText(input, {
    ...DEFAULT_TEXT_OPTIONS,
    // Allow some basic formatting for user input
    ALLOWED_TAGS: ['br', 'p'],
  });
};

/**
 * Sanitizes external data from APIs or other sources
 * @param data The external data to sanitize
 * @returns Sanitized data
 */
export const sanitizeExternalData = <T>(data: T): T => {
  if (typeof data === 'string') {
    return sanitizeText(data) as T;
  }
  
  if (Array.isArray(data)) {
    return data.map(item => sanitizeExternalData(item)) as T;
  }
  
  if (typeof data === 'object' && data !== null) {
    return sanitizeObject(data as Record<string, unknown>) as T;
  }
  
  return data;
};