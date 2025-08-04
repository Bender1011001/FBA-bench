"""
Unit tests for the error handler module.

This module tests the functionality of the error handler, including
the enhanced error handling for common agent errors.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import logging

from tools.error_handler import handle_common_errors_for_agent


class TestErrorHandler:
    """Test cases for the error handler module."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.agent_id = "test_agent"
        return agent

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock(spec=logging.Logger)

    def test_handle_common_errors_for_agent_with_value_error(self, mock_agent, mock_logger):
        """Test handling ValueError with proper logging."""
        error = ValueError("Test value error")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, mock_agent)
            
            # Should return None
            assert result is None
            
            # Should log the error
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "ValueError" in args[0]
            assert "Test value error" in args[0]
            assert "test_agent" in args[0]

    def test_handle_common_errors_for_agent_with_key_error(self, mock_agent, mock_logger):
        """Test handling KeyError with proper logging."""
        error = KeyError("missing_key")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, mock_agent)
            
            # Should return None
            assert result is None
            
            # Should log the error
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "KeyError" in args[0]
            assert "missing_key" in args[0]
            assert "test_agent" in args[0]

    def test_handle_common_errors_for_agent_with_type_error(self, mock_agent, mock_logger):
        """Test handling TypeError with proper logging."""
        error = TypeError("Test type error")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, mock_agent)
            
            # Should return None
            assert result is None
            
            # Should log the error
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "TypeError" in args[0]
            assert "Test type error" in args[0]
            assert "test_agent" in args[0]

    def test_handle_common_errors_for_agent_with_attribute_error(self, mock_agent, mock_logger):
        """Test handling AttributeError with proper logging."""
        error = AttributeError("Test attribute error")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, mock_agent)
            
            # Should return None
            assert result is None
            
            # Should log the error
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "AttributeError" in args[0]
            assert "Test attribute error" in args[0]
            assert "test_agent" in args[0]

    def test_handle_common_errors_for_agent_with_runtime_error(self, mock_agent, mock_logger):
        """Test handling RuntimeError with proper logging."""
        error = RuntimeError("Test runtime error")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, mock_agent)
            
            # Should return None
            assert result is None
            
            # Should log the error
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "RuntimeError" in args[0]
            assert "Test runtime error" in args[0]
            assert "test_agent" in args[0]

    def test_handle_common_errors_for_agent_with_unexpected_error(self, mock_agent, mock_logger):
        """Test handling unexpected errors with proper logging."""
        error = Exception("Unexpected error")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, mock_agent)
            
            # Should return None
            assert result is None
            
            # Should log the error
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "Unexpected error" in args[0]
            assert "Unexpected error" in args[0]
            assert "test_agent" in args[0]

    def test_handle_common_errors_for_agent_without_agent(self, mock_logger):
        """Test handling errors when no agent is provided."""
        error = ValueError("Test value error")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, None)
            
            # Should return None
            assert result is None
            
            # Should log the error without agent ID
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "ValueError" in args[0]
            assert "Test value error" in args[0]
            assert "unknown agent" in args[0]

    def test_handle_common_errors_for_agent_with_complex_error_message(self, mock_agent, mock_logger):
        """Test handling errors with complex error messages."""
        error = ValueError("Complex error with special chars: !@#$%^&*()")
        
        with patch('tools.error_handler.logger', mock_logger):
            result = handle_common_errors_for_agent(error, mock_agent)
            
            # Should return None
            assert result is None
            
            # Should log the error
            mock_logger.error.assert_called_once()
            args = mock_logger.error.call_args[0]
            assert "ValueError" in args[0]
            assert "Complex error with special chars: !@#$%^&*()" in args[0]
            assert "test_agent" in args[0]