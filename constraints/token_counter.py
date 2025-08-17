"""
Token Counter - Utility for counting tokens in text.

This module provides utilities for counting tokens in text, which is
essential for managing LLM API costs, context window limits, and
rate limiting.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Attempt to import tiktoken, make it an optional feature
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class TokenCountResult:
    """
    Result of token counting operation.
    
    Attributes:
        count: Number of tokens counted
        model: Model used for counting (if applicable)
        method: Method used for counting
        text_sample: Sample of text that was counted (truncated if too long)
        estimated: Whether the count is estimated or exact
    """
    count: int
    model: Optional[str] = None
    method: str = "unknown"
    text_sample: str = ""
    estimated: bool = False


class TokenCounter:
    """
    Utility class for counting tokens in text.
    
    This class provides methods to count tokens using various strategies,
    including exact counting with tiktoken (when available) and fallback
    estimation methods.
    """
    
    def __init__(self, default_model: str = "gpt-3.5-turbo"):
        """
        Initialize the token counter.
        
        Args:
            default_model: Default model to use for token counting
        """
        self.default_model = default_model
        self.encoding_cache: Dict[str, Any] = {}
        
        # Initialize tiktoken encoding if available
        self.encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(default_model)
                logger.info(f"Initialized tiktoken encoding for model: {default_model}")
            except KeyError:
                logger.warning(f"Unknown model for tiktoken: {default_model}, using cl100k_base as fallback")
                self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
        method: Optional[str] = None
    ) -> TokenCountResult:
        """
        Count tokens in the given text.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting (uses default if not provided)
            method: Method to use for counting ("auto", "tiktoken", "estimate")
            
        Returns:
            TokenCountResult with count and metadata
        """
        if not text:
            return TokenCountResult(count=0, method="empty", text_sample="")
        
        model = model or self.default_model
        
        # Determine method
        if method == "auto" or method is None:
            method = "tiktoken" if TIKTOKEN_AVAILABLE else "estimate"
        
        # Count tokens based on method
        if method == "tiktoken" and TIKTOKEN_AVAILABLE:
            return self._count_with_tiktoken(text, model)
        else:
            return self._estimate_tokens(text, model)
    
    def _count_with_tiktoken(self, text: str, model: str) -> TokenCountResult:
        """
        Count tokens using tiktoken.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting
            
        Returns:
            TokenCountResult with count and metadata
        """
        try:
            # Get or create encoding for the model
            encoding = self._get_encoding(model)
            
            # Count tokens
            tokens = encoding.encode(text)
            count = len(tokens)
            
            # Create text sample (first 100 chars)
            text_sample = text[:100] + "..." if len(text) > 100 else text
            
            return TokenCountResult(
                count=count,
                model=model,
                method="tiktoken",
                text_sample=text_sample,
                estimated=False
            )
            
        except Exception as e:
            logger.error(f"Error counting tokens with tiktoken: {e}")
            # Fall back to estimation
            return self._estimate_tokens(text, model)
    
    def _get_encoding(self, model: str):
        """
        Get or create tiktoken encoding for a model.
        
        Args:
            model: Model name
            
        Returns:
            Tiktoken encoding object
        """
        if model in self.encoding_cache:
            return self.encoding_cache[model]
        
        try:
            encoding = tiktoken.encoding_for_model(model)
            self.encoding_cache[model] = encoding
            return encoding
        except KeyError:
            logger.warning(f"Unknown model for tiktoken: {model}, using cl100k_base as fallback")
            encoding = tiktoken.get_encoding("cl100k_base")
            self.encoding_cache[model] = encoding
            return encoding
    
    def _estimate_tokens(self, text: str, model: str) -> TokenCountResult:
        """
        Estimate tokens using character-based heuristics.
        
        Args:
            text: Text to estimate tokens for
            model: Model to use for estimation (affects heuristic)
            
        Returns:
            TokenCountResult with estimated count and metadata
        """
        # Different models have different token-to-character ratios
        # These are rough estimates based on common patterns
        ratios = {
            "gpt-4": 0.25,      # ~4 chars per token
            "gpt-3.5-turbo": 0.25,  # ~4 chars per token
            "text-davinci-003": 0.25,  # ~4 chars per token
            "claude": 0.3,      # ~3.3 chars per token
            "default": 0.25    # Default to ~4 chars per token
        }
        
        # Get ratio for model or use default
        ratio = ratios.get(model, ratios["default"])
        
        # Calculate estimated token count
        count = int(len(text) * ratio)
        
        # Create text sample (first 100 chars)
        text_sample = text[:100] + "..." if len(text) > 100 else text
        
        return TokenCountResult(
            count=count,
            model=model,
            method="estimate",
            text_sample=text_sample,
            estimated=True
        )
    
    def count_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        method: Optional[str] = None
    ) -> TokenCountResult:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for counting
            method: Method to use for counting
            
        Returns:
            TokenCountResult with total count and metadata
        """
        if not messages:
            return TokenCountResult(count=0, method="empty", text_sample="")
        
        model = model or self.default_model
        
        # Determine method
        if method == "auto" or method is None:
            method = "tiktoken" if TIKTOKEN_AVAILABLE else "estimate"
        
        # Count tokens based on method
        if method == "tiktoken" and TIKTOKEN_AVAILABLE:
            return self._count_messages_with_tiktoken(messages, model)
        else:
            return self._estimate_message_tokens(messages, model)
    
    def _count_messages_with_tiktoken(self, messages: List[Dict[str, str]], model: str) -> TokenCountResult:
        """
        Count tokens in messages using tiktoken.
        
        Args:
            messages: List of message dictionaries
            model: Model to use for counting
            
        Returns:
            TokenCountResult with total count and metadata
        """
        # Get encoding for the model
        encoding = self._get_encoding(model)
        
        # Count tokens per message and add overhead
        # Based on OpenAI's token counting guidance
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        total_tokens = 0
        
        for message in messages:
            # Add message format tokens
            total_tokens += 4  # Each message adds ~4 tokens for format
            
            # Add content tokens
            content = message.get("content", "")
            if content:
                total_tokens += len(encoding.encode(content))
            
            # Add role tokens
            role = message.get("role", "")
            if role:
                total_tokens += len(encoding.encode(role))
        
        # Add final tokens for the assistant's reply
        total_tokens += 2  # Every reply is primed with <|start|>assistant<|message|>
        
        # Create text sample from first message
        text_sample = ""
        if messages:
            first_content = messages[0].get("content", "")
            text_sample = first_content[:100] + "..." if len(first_content) > 100 else first_content
        
        return TokenCountResult(
            count=total_tokens,
            model=model,
            method="tiktoken",
            text_sample=text_sample,
            estimated=False
        )
    
    def _estimate_message_tokens(self, messages: List[Dict[str, str]], model: str) -> TokenCountResult:
        """
        Estimate tokens in messages using character-based heuristics.
        
        Args:
            messages: List of message dictionaries
            model: Model to use for estimation
            
        Returns:
            TokenCountResult with estimated count and metadata
        """
        # Different models have different token-to-character ratios
        ratios = {
            "gpt-4": 0.25,      # ~4 chars per token
            "gpt-3.5-turbo": 0.25,  # ~4 chars per token
            "text-davinci-003": 0.25,  # ~4 chars per token
            "claude": 0.3,      # ~3.3 chars per token
            "default": 0.25    # Default to ~4 chars per token
        }
        
        # Get ratio for model or use default
        ratio = ratios.get(model, ratios["default"])
        
        # Calculate estimated token count
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = int(total_chars * ratio)
        
        # Add overhead for message formatting (rough estimate)
        # Each message adds some overhead for role and formatting
        estimated_tokens += len(messages) * 6  # ~6 tokens per message for format
        
        # Add final tokens for the assistant's reply
        estimated_tokens += 2  # Every reply is primed with some tokens
        
        # Create text sample from first message
        text_sample = ""
        if messages:
            first_content = messages[0].get("content", "")
            text_sample = first_content[:100] + "..." if len(first_content) > 100 else first_content
        
        return TokenCountResult(
            count=estimated_tokens,
            model=model,
            method="estimate",
            text_sample=text_sample,
            estimated=True
        )
    
    def count_tokens_by_chunks(
        self,
        text: str,
        chunk_size: int = 1000,
        model: Optional[str] = None,
        method: Optional[str] = None
    ) -> List[TokenCountResult]:
        """
        Count tokens in text by processing it in chunks.
        
        Args:
            text: Text to count tokens for
            chunk_size: Size of each chunk in characters
            model: Model to use for counting
            method: Method to use for counting
            
        Returns:
            List of TokenCountResult objects, one for each chunk
        """
        if not text:
            return []
        
        model = model or self.default_model
        
        # Split text into chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Count tokens for each chunk
        results = []
        for i, chunk in enumerate(chunks):
            result = self.count_tokens(chunk, model, method)
            result.text_sample = f"Chunk {i+1}: {result.text_sample}"
            results.append(result)
        
        return results
    
    def get_token_usage_stats(
        self,
        results: List[TokenCountResult]
    ) -> Dict[str, Any]:
        """
        Get statistics from a list of token count results.
        
        Args:
            results: List of TokenCountResult objects
            
        Returns:
            Dictionary with usage statistics
        """
        if not results:
            return {
                "total_tokens": 0,
                "average_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "estimated_count": 0,
                "exact_count": 0,
                "models_used": [],
                "methods_used": []
            }
        
        total_tokens = sum(r.count for r in results)
        average_tokens = total_tokens / len(results)
        min_tokens = min(r.count for r in results)
        max_tokens = max(r.count for r in results)
        estimated_count = sum(1 for r in results if r.estimated)
        exact_count = sum(1 for r in results if not r.estimated)
        models_used = list(set(r.model for r in results if r.model))
        methods_used = list(set(r.method for r in results))
        
        return {
            "total_tokens": total_tokens,
            "average_tokens": average_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "estimated_count": estimated_count,
            "exact_count": exact_count,
            "models_used": models_used,
            "methods_used": methods_used
        }
    
    def is_available(self) -> bool:
        """
        Check if tiktoken is available for exact token counting.
        
        Returns:
            True if tiktoken is available, False otherwise
        """
        return TIKTOKEN_AVAILABLE
