// frontend/src/utils/validators.ts

interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  min?: number;
  max?: number;
  type?: 'string' | 'number' | 'integer' | 'boolean';
  enum?: string[];
  pattern?: RegExp;
  custom?: (value: unknown) => string | undefined; // Returns error message or undefined
}

type ValidationSchema<T> = {
  [K in keyof T]?: ValidationRule;
};

// Generic validation function
export const validateField = (value: unknown, rule?: ValidationRule): string | undefined => {
  if (!rule) {
    return undefined; // No validation rule, so no error
  }

  // Required field validation
  if (rule.required && (value === null || value === undefined || value === '')) {
    return 'This field is required.';
  }

  // If not required and value is empty, no further validation for this field
  if (!rule.required && (value === null || value === undefined || value === '')) {
    return undefined;
  }

  // Type validation
  if (rule.type) {
    if (rule.type === 'number') {
      if (typeof value !== 'number' || isNaN(value)) {
        return 'Must be a valid number.';
      }
    } else if (rule.type === 'integer') {
      if (typeof value !== 'number' || !Number.isInteger(value)) {
        return 'Must be a valid integer.';
      }
    } else if (rule.type === 'string') {
      if (typeof value !== 'string') {
        return 'Must be a string.';
      }
    } else if (rule.type === 'boolean') {
      if (typeof value !== 'boolean') {
        return 'Must be a boolean.';
      }
    }
  }

  // Length validation (for strings)
  if (typeof value === 'string') {
    if (rule.minLength !== undefined && value.length < rule.minLength) {
      return `Minimum length is ${rule.minLength} characters.`;
    }
    if (rule.maxLength !== undefined && value.length > rule.maxLength) {
      return `Maximum length is ${rule.maxLength} characters.`;
    }
  }

  // Range validation (for numbers)
  if (typeof value === 'number') {
    if (rule.min !== undefined && value < rule.min) {
      return `Minimum value is ${rule.min}.`;
    }
    if (rule.max !== undefined && value > rule.max) {
      return `Maximum value is ${rule.max}.`;
    }
  }

  // Enum validation
  if (rule.enum !== undefined && !rule.enum.includes(String(value))) {
    return `Value must be one of: ${rule.enum.join(', ')}.`;
  }

  // Pattern validation (for strings)
  if (typeof value === 'string' && rule.pattern) {
    if (!rule.pattern.test(value)) {
      return 'Invalid format.';
    }
  }

  // Custom validation
  if (rule.custom) {
    return rule.custom(value);
  }

  return undefined; // No errors
};

// Generic function to validate an entire form object
export const validateForm = <T extends Record<string, unknown>>(
  formData: T,
  schema: ValidationSchema<T>
): Record<keyof T, string | undefined> => {
  const errors: Record<keyof T, string | undefined> = {} as Record<keyof T, string | undefined>;

  for (const key in schema) {
    if (Object.prototype.hasOwnProperty.call(schema, key)) {
      const rule = schema[key];
      const value = formData[key];
      errors[key] = validateField(value, rule);
    }
  }
  return errors;
};

// Specific validation schemas based on user requirements

// Simulation validation
export interface SimulationValidationRules {
  simulationName: { required: true, minLength: 3, maxLength: 100, type: 'string' };
  duration: { required: true, min: 1, max: 168, type: 'number' };
  tickInterval: { required: true, min: 1, max: 3600, type: 'number' };
  initialPrice: { required: true, min: 0.01, max: 10000, type: 'number' };
  inventory: { required: true, min: 0, max: 1000000, type: 'integer' };
}

export const simulationValidationSchema: ValidationSchema<SimulationValidationRules> = {
  simulationName: { required: true, minLength: 3, maxLength: 100, type: 'string' },
  duration: { required: true, min: 1, max: 168, type: 'number' },
  tickInterval: { required: true, min: 1, max: 3600, type: 'number' },
  initialPrice: { required: true, min: 0.01, max: 10000, type: 'number' },
  inventory: { required: true, min: 0, max: 1000000, type: 'integer' },
};

// Agent validation
export interface AgentFormData {
  framework: string;
  llmModel: string;
  temperature: number;
  maxTokens: number;
  [key: string]: string | number; // Add index signature for form flexibility
}

export interface AgentValidationRules {
  framework: { required: true, enum: ['DIY', 'CrewAI', 'LangChain'], type: 'string' };
  llmModel: { required: true, type: 'string' };
  temperature: { min: 0, max: 2, type: 'number' };
  maxTokens: { required: true, min: 1, max: 32000, type: 'integer' }; // Made required based on form
}

export const agentValidationSchema: ValidationSchema<AgentValidationRules> = {
  framework: { required: true, enum: ['DIY', 'CrewAI', 'LangChain'], type: 'string' },
  llmModel: { required: true, type: 'string' },
  temperature: { min: 0, max: 2, type: 'number' },
  maxTokens: { required: true, min: 1, max: 32000, type: 'integer' },
};


// Constraint validation
export interface ConstraintValidationRules {
  budgetLimit: { required: true, min: 1, max: 10000, type: 'number' };
  tokenLimit: { required: true, min: 100, max: 1000000, type: 'integer' };
  rateLimit: { required: true, min: 1, max: 1000, type: 'integer' };
}

export const constraintValidationSchema: ValidationSchema<ConstraintValidationRules> = {
  budgetLimit: { required: true, min: 1, max: 10000, type: 'number' },
  tokenLimit: { required: true, min: 100, max: 1000000, type: 'integer' },
  rateLimit: { required: true, min: 1, max: 1000, type: 'integer' },
};