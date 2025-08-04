import React from 'react';
import { render, screen } from '@testing-library/react';
import ErrorBoundary from '../ErrorBoundary';

// Mock console.error to avoid noise in test output
const originalError = console.error;
beforeEach(() => {
  console.error = jest.fn();
});

afterEach(() => {
  console.error = originalError;
});

describe('ErrorBoundary Component', () => {
  test('renders children when there is no error', () => {
    const ChildComponent = () => <div>Child Component</div>;
    
    render(
      <ErrorBoundary>
        <ChildComponent />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Child Component')).toBeInTheDocument();
  });

  test('renders error UI when there is an error', () => {
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('Test error')).toBeInTheDocument();
    expect(screen.getByText('Try again')).toBeInTheDocument();
  });

  test('calls onError callback when an error occurs', () => {
    const onError = jest.fn();
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    render(
      <ErrorBoundary onError={onError}>
        <ThrowError />
      </ErrorBoundary>
    );
    
    expect(onError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.any(Object)
    );
  });

  test('logs error to console', () => {
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );
    
    expect(console.error).toHaveBeenCalledWith(
      'Error caught by ErrorBoundary:',
      expect.any(Error),
      expect.any(Object)
    );
  });

  test('renders custom fallback when provided', () => {
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    const CustomFallback = () => <div>Custom Error UI</div>;
    
    render(
      <ErrorBoundary fallback={<CustomFallback />}>
        <ThrowError />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Custom Error UI')).toBeInTheDocument();
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
  });

  test('resets error state when reset button is clicked', () => {
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    const { rerender } = render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    
    // Click the reset button
    screen.getByText('Try again').click();
    
    // Render with a component that doesn't throw
    const SafeComponent = () => <div>Safe Component</div>;
    
    rerender(
      <ErrorBoundary>
        <SafeComponent />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Safe Component')).toBeInTheDocument();
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
  });

  test('shows error details in development mode', () => {
    // Set NODE_ENV to development
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';
    
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Error details (development mode)')).toBeInTheDocument();
    
    // Restore original NODE_ENV
    process.env.NODE_ENV = originalEnv;
  });

  test('does not show error details in production mode', () => {
    // Set NODE_ENV to production
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'production';
    
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );
    
    expect(screen.queryByText('Error details (development mode)')).not.toBeInTheDocument();
    
    // Restore original NODE_ENV
    process.env.NODE_ENV = originalEnv;
  });

  test('handles errors in nested components', () => {
    const ParentComponent = () => (
      <div>
        <h1>Parent Component</h1>
        <ChildComponent />
      </div>
    );
    
    const ChildComponent = () => {
      throw new Error('Child error');
    };
    
    render(
      <ErrorBoundary>
        <ParentComponent />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('Child error')).toBeInTheDocument();
    expect(screen.queryByText('Parent Component')).not.toBeInTheDocument();
  });

  test('handles errors in multiple children', () => {
    const FirstChild = () => <div>First Child</div>;
    const SecondChild = () => {
      throw new Error('Second child error');
    };
    
    render(
      <ErrorBoundary>
        <FirstChild />
        <SecondChild />
      </ErrorBoundary>
    );
    
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('Second child error')).toBeInTheDocument();
    expect(screen.queryByText('First Child')).not.toBeInTheDocument();
  });
});