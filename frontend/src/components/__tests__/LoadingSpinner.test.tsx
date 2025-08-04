import React from 'react';
import { render, screen } from '@testing-library/react';
import LoadingSpinner from '../LoadingSpinner';

describe('LoadingSpinner Component', () => {
  test('renders without crashing', () => {
    render(<LoadingSpinner />);
    expect(screen.getByRole('img')).toBeInTheDocument();
  });

  test('has correct default size', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('img');
    expect(spinner).toHaveClass('h-8 w-8');
  });

  test('has correct default color', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('img');
    expect(spinner).toHaveClass('text-blue-600');
  });

  test('applies small size when specified', () => {
    render(<LoadingSpinner size="small" />);
    const spinner = screen.getByRole('img');
    expect(spinner).toHaveClass('h-4 w-4');
  });

  test('applies medium size when specified', () => {
    render(<LoadingSpinner size="medium" />);
    const spinner = screen.getByRole('img');
    expect(spinner).toHaveClass('h-8 w-8');
  });

  test('applies large size when specified', () => {
    render(<LoadingSpinner size="large" />);
    const spinner = screen.getByRole('img');
    expect(spinner).toHaveClass('h-12 w-12');
  });

  test('applies custom color when specified', () => {
    render(<LoadingSpinner color="text-red-500" />);
    const spinner = screen.getByRole('img');
    expect(spinner).toHaveClass('text-red-500');
  });

  test('applies custom className when specified', () => {
    render(<LoadingSpinner className="custom-class" />);
    const container = screen.getByRole('img').parentElement;
    expect(container).toHaveClass('custom-class');
  });

  test('has correct SVG structure', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('img');
    
    expect(spinner).toHaveAttribute('xmlns', 'http://www.w3.org/2000/svg');
    expect(spinner).toHaveAttribute('fill', 'none');
    expect(spinner).toHaveAttribute('viewBox', '0 0 24 24');
    
    // Check for circle element
    const circle = spinner.querySelector('circle');
    expect(circle).toBeInTheDocument();
    expect(circle).toHaveAttribute('cx', '12');
    expect(circle).toHaveAttribute('cy', '12');
    expect(circle).toHaveAttribute('r', '10');
    expect(circle).toHaveAttribute('stroke', 'currentColor');
    expect(circle).toHaveAttribute('strokeWidth', '4');
    
    // Check for path element
    const path = spinner.querySelector('path');
    expect(path).toBeInTheDocument();
    expect(path).toHaveAttribute('fill', 'currentColor');
    expect(path).toHaveAttribute('d', 'M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z');
  });

  test('has animation class', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('img');
    expect(spinner).toHaveClass('animate-spin');
  });

  test('has opacity classes for animation effect', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('img');
    
    const circle = spinner.querySelector('circle');
    expect(circle).toHaveClass('opacity-25');
    
    const path = spinner.querySelector('path');
    expect(path).toHaveClass('opacity-75');
  });
});