"""
Unit tests for the LLM cache module.

This module tests the functionality of the LLM cache, including
the enhanced error handling in the cleanup logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import logging
import tempfile
import os

from reproducibility.llm_cache import LLMPredictionCache


class TestLLMCache:
    """Test cases for the LLM cache module."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def cache_config(self, temp_dir):
        """Create a cache config for testing."""
        return {
            "cache_dir": temp_dir,
            "max_size_mb": 100,
            "ttl_seconds": 3600
        }

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def cache(self, cache_config):
        """Create an LLMPredictionCache instance for testing."""
        return LLMPredictionCache(cache_config)

    def test_initialization(self, cache, cache_config):
        """Test that the cache initializes correctly."""
        assert cache.cache_dir == cache_config["cache_dir"]
        assert cache.max_size_mb == cache_config["max_size_mb"]
        assert cache.ttl_seconds == cache_config["ttl_seconds"]

    @pytest.mark.asyncio
    async def test_close_success(self, cache, mock_logger):
        """Test successful cache closing."""
        with patch('reproducibility.llm_cache.logger', mock_logger):
            await cache._close()
            
            # Should not log any errors
            mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_exception(self, cache, mock_logger):
        """Test cache closing with an exception."""
        # Mock the close method to raise an exception
        with patch.object(cache, '_close', side_effect=Exception("Close error")):
            with patch('reproducibility.llm_cache.logger', mock_logger):
                await cache._close()
                
                # Should log the error
                mock_logger.error.assert_called_once()
                args = mock_logger.error.call_args[0]
                assert "Error closing LLM cache" in args[0]
                assert "Close error" in args[0]

    @pytest.mark.asyncio
    async def test_context_manager_success(self, cache_config, mock_logger):
        """Test using the cache as a context manager successfully."""
        with patch('reproducibility.llm_cache.logger', mock_logger):
            async with LLMPredictionCache(cache_config) as cache:
                assert isinstance(cache, LLMPredictionCache)
            
            # Should not log any errors
            mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception_in_close(self, cache_config, mock_logger):
        """Test using the cache as a context manager with an exception in close."""
        with patch('reproducibility.llm_cache.logger', mock_logger):
            # Create a cache that will raise an exception when closed
            cache = LLMPredictionCache(cache_config)
            
            # Mock the _close method to raise an exception
            with patch.object(cache, '_close', side_effect=Exception("Context close error")):
                try:
                    async with cache:
                        pass
                except Exception:
                    pass
                
                # Should log the error
                mock_logger.error.assert_called_once()
                args = mock_logger.error.call_args[0]
                assert "Error closing LLM cache" in args[0]
                assert "Context close error" in args[0]

    @pytest.mark.asyncio
    async def test_context_manager_with_exception_in_body(self, cache_config, mock_logger):
        """Test using the cache as a context manager with an exception in the body."""
        with patch('reproducibility.llm_cache.logger', mock_logger):
            try:
                async with LLMPredictionCache(cache_config) as cache:
                    raise ValueError("Body error")
            except ValueError:
                pass
            
            # Should not log any errors related to cache closing
            # The body error is not the responsibility of the cache
            mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_close_calls(self, cache, mock_logger):
        """Test calling close multiple times."""
        with patch('reproducibility.llm_cache.logger', mock_logger):
            # Close the cache multiple times
            await cache._close()
            await cache._close()
            await cache._close()
            
            # Should not log any errors
            mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_complex_exception(self, cache, mock_logger):
        """Test cache closing with a complex exception."""
        # Create a complex exception with nested exceptions
        inner_error = ValueError("Inner error")
        outer_error = RuntimeError("Outer error", inner_error)
        
        # Mock the close method to raise the complex exception
        with patch.object(cache, '_close', side_effect=outer_error):
            with patch('reproducibility.llm_cache.logger', mock_logger):
                await cache._close()
                
                # Should log the error
                mock_logger.error.assert_called_once()
                args = mock_logger.error.call_args[0]
                assert "Error closing LLM cache" in args[0]
                assert "Outer error" in args[0]