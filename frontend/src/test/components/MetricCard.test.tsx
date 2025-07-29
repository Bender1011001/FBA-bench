import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricCard, MetricCardSkeleton, MetricCardError } from '../../components/MetricCard';
import type { DashboardMetric } from '../../types';

const mockMetric: DashboardMetric = {
  label: 'Total Sales',
  value: 1234.56,
  formatType: 'currency',
  trend: 'up'
};

describe('MetricCard', () => {
  describe('Basic Rendering', () => {
    it('should render metric label and value', () => {
      render(<MetricCard metric={mockMetric} />);
      
      expect(screen.getByText('Total Sales')).toBeInTheDocument();
      expect(screen.getByText('$1,234.56')).toBeInTheDocument();
    });

    it('should apply custom className', () => {
      const { container } = render(
        <MetricCard metric={mockMetric} className="custom-class" />
      );
      
      expect(container.firstChild).toHaveClass('custom-class');
    });
  });

  describe('Value Formatting', () => {
    it('should format currency values correctly', () => {
      const metric = { ...mockMetric, value: 1000, formatType: 'currency' as const };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('$1,000.00')).toBeInTheDocument();
    });

    it('should format percentage values correctly', () => {
      const metric = { ...mockMetric, value: 85, formatType: 'percentage' as const };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('85.0%')).toBeInTheDocument();
    });

    it('should format number values correctly', () => {
      const metric = { ...mockMetric, value: 12345, formatType: 'number' as const };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('12,345')).toBeInTheDocument();
    });

    it('should handle string values', () => {
      const metric = { ...mockMetric, value: 'Test String' };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('Test String')).toBeInTheDocument();
    });

    it('should handle values without formatType', () => {
      const metric = { ...mockMetric, value: 42, formatType: undefined };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('42')).toBeInTheDocument();
    });
  });

  describe('Trend Indicators', () => {
    it('should show up trend icon and text', () => {
      const metric = { ...mockMetric, trend: 'up' as const };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('Trending up')).toBeInTheDocument();
      // SVG should be present
      expect(document.querySelector('svg')).toBeInTheDocument();
    });

    it('should show down trend icon and text', () => {
      const metric = { ...mockMetric, trend: 'down' as const };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('Trending down')).toBeInTheDocument();
      expect(document.querySelector('svg')).toBeInTheDocument();
    });

    it('should show neutral trend icon and text', () => {
      const metric = { ...mockMetric, trend: 'neutral' as const };
      render(<MetricCard metric={metric} />);
      
      expect(screen.getByText('No change')).toBeInTheDocument();
      expect(document.querySelector('svg')).toBeInTheDocument();
    });

    it('should not show trend section when no trend provided', () => {
      const metric = { ...mockMetric, trend: undefined };
      render(<MetricCard metric={metric} />);
      
      expect(screen.queryByText('Trending up')).not.toBeInTheDocument();
      expect(screen.queryByText('Trending down')).not.toBeInTheDocument();
      expect(screen.queryByText('No change')).not.toBeInTheDocument();
    });
  });

  describe('CSS Classes', () => {
    it('should apply correct color classes for up trend', () => {
      const metric = { ...mockMetric, trend: 'up' as const };
      const { container } = render(<MetricCard metric={metric} />);
      
      expect(container.querySelector('.text-green-600')).toBeInTheDocument();
    });

    it('should apply correct color classes for down trend', () => {
      const metric = { ...mockMetric, trend: 'down' as const };
      const { container } = render(<MetricCard metric={metric} />);
      
      expect(container.querySelector('.text-red-600')).toBeInTheDocument();
    });

    it('should apply correct color classes for neutral trend', () => {
      const metric = { ...mockMetric, trend: 'neutral' as const };
      const { container } = render(<MetricCard metric={metric} />);
      
      expect(container.querySelector('.text-gray-600')).toBeInTheDocument();
    });

    it('should apply default color classes for no trend', () => {
      const metric = { ...mockMetric, trend: undefined };
      const { container } = render(<MetricCard metric={metric} />);
      
      expect(container.querySelector('.text-gray-900')).toBeInTheDocument();
    });
  });
});

describe('MetricCardSkeleton', () => {
  it('should render skeleton loader', () => {
    const { container } = render(<MetricCardSkeleton />);
    
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
    expect(container.querySelector('.bg-gray-200')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    const { container } = render(<MetricCardSkeleton className="custom-skeleton" />);
    
    expect(container.firstChild).toHaveClass('custom-skeleton');
  });
});

describe('MetricCardError', () => {
  it('should render error state with label and error message', () => {
    render(<MetricCardError label="Test Metric" error="Network error" />);
    
    expect(screen.getByText('Test Metric')).toBeInTheDocument();
    expect(screen.getByText('Error: Network error')).toBeInTheDocument();
    expect(screen.getByText('--')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    const { container } = render(
      <MetricCardError 
        label="Test Metric" 
        error="Network error" 
        className="custom-error" 
      />
    );
    
    expect(container.firstChild).toHaveClass('custom-error');
  });

  it('should show error icon', () => {
    render(<MetricCardError label="Test Metric" error="Network error" />);
    
    expect(document.querySelector('svg')).toBeInTheDocument();
    expect(document.querySelector('.text-red-500')).toBeInTheDocument();
  });

  it('should have red border styling', () => {
    const { container } = render(
      <MetricCardError label="Test Metric" error="Network error" />
    );
    
    expect(container.querySelector('.border-red-200')).toBeInTheDocument();
  });
});

describe('MetricCard Edge Cases', () => {
  it('should handle very large numbers', () => {
    const metric = { 
      ...mockMetric, 
      value: 999999999, 
      formatType: 'number' as const 
    };
    render(<MetricCard metric={metric} />);
    
    expect(screen.getByText('999,999,999')).toBeInTheDocument();
  });

  it('should handle zero values', () => {
    const metric = { ...mockMetric, value: 0 };
    render(<MetricCard metric={metric} />);
    
    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('should handle negative numbers with currency format', () => {
    const metric = { 
      ...mockMetric, 
      value: -500, 
      formatType: 'currency' as const 
    };
    render(<MetricCard metric={metric} />);
    
    expect(screen.getByText('-$500.00')).toBeInTheDocument();
  });

  it('should handle decimal percentages', () => {
    const metric = { 
      ...mockMetric, 
      value: 0.567, 
      formatType: 'percentage' as const 
    };
    render(<MetricCard metric={metric} />);
    
    expect(screen.getByText('0.6%')).toBeInTheDocument();
  });
});