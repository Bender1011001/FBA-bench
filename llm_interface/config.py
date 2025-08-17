"""
Configuration classes for Language Model (LLM) interfaces in FBA-Bench.

This module defines the `LLMConfig` dataclass, which is used to specify
the configuration parameters for interacting with various LLM providers and models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class LLMConfig:
    """
    Configuration for an LLM provider and model.
    """
    provider: str  # e.g., "openai", "openrouter", "google"
    model: str     # e.g., "gpt-4o-mini", "google/gemini-flash-2.5"
    api_key_env: Optional[str] = None  # Environment variable name for API key
    base_url: Optional[str] = None     # Custom base URL for API endpoint
    temperature: float = 0.7           # Creativity/randomness of output
    max_tokens: int = 1024             # Maximum tokens in the response
    top_p: float = 1.0                 # Nucleus sampling parameter
    frequency_penalty: float = 0.0     # Penalty for new tokens based on their frequency
    presence_penalty: float = 0.0      # Penalty for new tokens based on whether they appear in the text so far
    timeout: int = 60                  # Request timeout in seconds
    max_retries: int = 3               # Maximum number of retries for failed requests
    
    # Provider-specific parameters that don't fit into generic fields
    custom_params: Dict[str, Any] = field(default_factory=dict)