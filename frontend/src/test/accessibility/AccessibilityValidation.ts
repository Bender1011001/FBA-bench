/**
 * Accessibility Validation Utilities
 * 
 * This file provides utilities for validating accessibility features
 * across the benchmarking components without external dependencies.
 */

// Accessibility validation rules and checks
export interface AccessibilityRule {
  id: string;
  name: string;
  description: string;
  check: (element: HTMLElement) => boolean;
  errorMessage: string;
}

// Common accessibility rules
export const ACCESSIBILITY_RULES: AccessibilityRule[] = [
  {
    id: 'has-aria-label',
    name: 'Has ARIA Label',
    description: 'Interactive elements should have proper ARIA labels',
    check: (element) => {
      return element.hasAttribute('aria-label') || 
             element.hasAttribute('aria-labelledby') ||
             element.hasAttribute('title');
    },
    errorMessage: 'Interactive element should have an aria-label, aria-labelledby, or title attribute'
  },
  {
    id: 'has-tabindex',
    name: 'Has Tab Index',
    description: 'Interactive elements should be keyboard accessible',
    check: (element) => {
      return element.hasAttribute('tabindex') || 
             element.tagName === 'BUTTON' ||
             element.tagName === 'A' ||
             element.tagName === 'INPUT' ||
             element.tagName === 'SELECT' ||
             element.tagName === 'TEXTAREA';
    },
    errorMessage: 'Interactive element should have a tabindex attribute or be a native interactive element'
  },
  {
    id: 'has-role',
    name: 'Has Proper Role',
    description: 'Elements should have appropriate ARIA roles',
    check: (element) => {
      const interactiveTags = ['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA', 'ROLE'];
      return interactiveTags.includes(element.tagName) || 
             element.hasAttribute('role');
    },
    errorMessage: 'Interactive element should have a proper role attribute'
  },
  {
    id: 'has-alt-text',
    name: 'Has Alt Text',
    description: 'Images should have alternative text',
    check: (element) => {
      return element.tagName !== 'IMG' || 
             element.hasAttribute('alt') && 
             element.getAttribute('alt')?.trim() !== '';
    },
    errorMessage: 'Images should have meaningful alt text'
  },
  {
    id: 'has-language',
    name: 'Has Language Attribute',
    description: 'HTML should have lang attribute',
    check: (element) => {
      return element.hasAttribute('lang') && 
             element.getAttribute('lang')?.trim() !== '';
    },
    errorMessage: 'HTML element should have a valid lang attribute'
  },
  {
    id: 'has-title',
    name: 'Has Title for Links',
    description: 'Links should have descriptive titles',
    check: (element) => {
      return element.tagName !== 'A' || 
             element.hasAttribute('title') && 
             element.getAttribute('title')?.trim() !== '';
    },
    errorMessage: 'Links should have descriptive title attributes'
  },
  {
    id: 'form-labels',
    name: 'Form Labels',
    description: 'Form inputs should have associated labels',
    check: (element) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName)) {
        const id = element.getAttribute('id');
        if (id) {
          const label = document.querySelector(`label[for="${id}"]`);
          return label !== null;
        }
      }
      return true;
    },
    errorMessage: 'Form inputs should have associated labels'
  },
  {
    id: 'focus-visible',
    name: 'Focus Visible',
    description: 'Elements should have visible focus indicators',
    check: (element) => {
      const style = window.getComputedStyle(element);
      return style.outline !== 'none' || 
             style.boxShadow !== 'none' ||
             style.border !== 'none';
    },
    errorMessage: 'Elements should have visible focus indicators'
  }
];

/**
 * Validate accessibility of a component
 */
export function validateAccessibility(element: HTMLElement): {
  isValid: boolean;
  violations: string[];
  rules: { rule: AccessibilityRule; passed: boolean }[];
} {
  const violations: string[] = [];
  const rules = ACCESSIBILITY_RULES.map(rule => {
    const passed = rule.check(element);
    if (!passed) {
      violations.push(rule.errorMessage);
    }
    return { rule, passed };
  });

  return {
    isValid: violations.length === 0,
    violations,
    rules
  };
}

/**
 * Validate keyboard navigation
 */
export function validateKeyboardNavigation(element: HTMLElement): {
  isValid: boolean;
  issues: string[];
} {
  const issues: string[] = [];
  
  // Check if element can receive focus
  if (!element.hasAttribute('tabindex') && 
      !['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName)) {
    issues.push('Element cannot receive keyboard focus');
  }

  // Check if element has visible focus indicator
  const style = window.getComputedStyle(element);
  if (style.outline === 'none' && style.boxShadow === 'none') {
    issues.push('Element has no visible focus indicator');
  }

  return {
    isValid: issues.length === 0,
    issues
  };
}

/**
 * Validate color contrast
 */
export function validateColorContrast(element: HTMLElement): {
  isValid: boolean;
  issues: string[];
} {
  const issues: string[] = [];
  
  try {
    const style = window.getComputedStyle(element);
    const color = style.color;
    const backgroundColor = style.backgroundColor;
    
    // Basic contrast check (simplified)
    if (color === 'rgb(255, 255, 255)' && backgroundColor === 'rgb(255, 255, 255)') {
      issues.push('Text color and background color are the same');
    }
    
    if (color === 'rgb(0, 0, 0)' && backgroundColor === 'rgb(0, 0, 0)') {
      issues.push('Text color and background color are the same');
    }
  } catch {
    issues.push('Could not validate color contrast');
  }

  return {
    isValid: issues.length === 0,
    issues
  };
}

/**
 * Generate accessibility report
 */
export function generateAccessibilityReport(componentName: string, element: HTMLElement): {
  component: string;
  timestamp: string;
  overallScore: number;
  rules: { rule: string; passed: boolean; message: string }[];
  keyboardNavigation: { valid: boolean; issues: string[] };
  colorContrast: { valid: boolean; issues: string[] };
} {
  const validation = validateAccessibility(element);
  const keyboardValidation = validateKeyboardNavigation(element);
  const contrastValidation = validateColorContrast(element);

  const overallScore = Math.round(
    (validation.rules.filter(r => r.passed).length / validation.rules.length) * 100
  );

  return {
    component: componentName,
    timestamp: new Date().toISOString(),
    overallScore,
    rules: validation.rules.map(r => ({
      rule: r.rule.name,
      passed: r.passed,
      message: r.passed ? 'Passed' : r.rule.errorMessage
    })),
    keyboardNavigation: {
      valid: keyboardValidation.isValid,
      issues: keyboardValidation.issues
    },
    colorContrast: {
      valid: contrastValidation.isValid,
      issues: contrastValidation.issues
    }
  };
}

/**
 * Accessibility testing utilities
 */
export interface AccessibilityReport {
  component: string;
  timestamp: string;
  overallScore: number;
  rules: { rule: string; passed: boolean; message: string }[];
  keyboardNavigation: { valid: boolean; issues: string[] };
  colorContrast: { valid: boolean; issues: string[] };
}

export class AccessibilityTester {
  private results: Map<string, AccessibilityReport> = new Map();

  /**
   * Test a component for accessibility
   */
  testComponent(componentName: string, element: HTMLElement): void {
    const report = generateAccessibilityReport(componentName, element);
    this.results.set(componentName, report);
  }

  /**
   * Get all test results
   */
  getResults(): Map<string, AccessibilityReport> {
    return new Map(this.results);
  }

  /**
   * Get overall accessibility score
   */
  getOverallScore(): number {
    if (this.results.size === 0) return 0;
    
    const totalScore = Array.from(this.results.values())
      .reduce((sum, report) => sum + report.overallScore, 0);
    
    return Math.round(totalScore / this.results.size);
  }

  /**
   * Generate summary report
   */
  generateSummary(): {
    totalComponents: number;
    overallScore: number;
    passedComponents: string[];
    failedComponents: string[];
  } {
    const passedComponents: string[] = [];
    const failedComponents: string[] = [];

    this.results.forEach((report, componentName) => {
      if (report.overallScore >= 90) {
        passedComponents.push(componentName);
      } else {
        failedComponents.push(componentName);
      }
    });

    return {
      totalComponents: this.results.size,
      overallScore: this.getOverallScore(),
      passedComponents,
      failedComponents
    };
  }
}

// Export default instance
export const accessibilityTester = new AccessibilityTester();