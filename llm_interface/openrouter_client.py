import os
import httpx
import tiktoken
import logging
from typing import Dict, Any, Optional

from llm_interface.contract import BaseLLMClient, LLMClientError

logger = logging.getLogger(__name__)

class OpenRouterClient(BaseLLMClient):
    """
    LLM client for OpenRouter API, adhering to BaseLLMClient interface.
    Uses httpx for asynchronous requests and tiktoken for token counting.
    """
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(model_name, api_key, base_url)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API Key not provided and not found in environment variables (OPENROUTER_API_KEY).")
        
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.http_client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

        # Initialize tiktoken for token counting (compatible with OpenAI models)
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            logger.warning(f"Could not find tiktoken encoding for model '{self.model_name}'. Using 'cl100k_base' as fallback. Token counts might be inaccurate.")
            self.encoding = tiktoken.get_encoding("cl100k_base")


    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = {"type": "json_object"}, # Default for structured output
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a structured response from the LLM based on the given prompt using OpenRouter API.

        Args:
            prompt: The formatted prompt string.
            temperature: Sampling temperature for the model.
            max_tokens: Maximum tokens to generate in the completion.
            response_format: Specifies the format of the response (e.g., {"type": "json_object"}).
            **kwargs: Additional parameters to pass to the OpenRouter API (e.g., top_p, frequency_penalty).

        Returns:
            A dictionary containing the LLM's raw response, similar to OpenAI's chat completions format.

        Raises:
            LLMClientError: If there is an issue communicating with the LLM API or receiving a valid response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://fba-bench.com", # Replace with your app's base URL
            "X-Request-Id": kwargs.pop("request_id", "") # Optional: for tracing requests
        }

        # OpenRouter API expects messages array for chat completions
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            **kwargs
        }

        try:
            logger.debug(f"Calling OpenRouter API for model {self.model_name} with payload: {payload}")
            response = await self.http_client.post(
                "/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            response_data = response.json()
            logger.debug(f"Received response from OpenRouter: {response_data}")

            # Basic validation of response structure comparable to OpenAI
            if not response_data.get("choices") or not isinstance(response_data["choices"], list):
                raise LLMClientError(f"OpenRouter response missing 'choices' field or not a list: {response_data}")
            if not response_data["choices"][0].get("message") or not response_data["choices"][0]["message"].get("content"):
                raise LLMClientError(f"OpenRouter response missing message content in choices: {response_data}")
            
            return response_data

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error connecting to OpenRouter API: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(
                f"OpenRouter API returned an HTTP error: {e.response.status_code} - {e.response.text}",
                original_exception=e,
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            logger.error(f"Network error connecting to OpenRouter API: {e}")
            raise LLMClientError(
                f"Network error connecting to OpenRouter API: {e}",
                original_exception=e
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenRouter API call: {e}", exc_info=True)
            raise LLMClientError(
                f"An unexpected error occurred during OpenRouter API call: {e}",
                original_exception=e
            )

    async def get_token_count(self, text: str) -> int:
        """
        Calculates the token count for a given text using tiktoken.
        """
        # tiktoken requires encoding text, not messages.
        # For actual prompt/completion token calculation, typically the OpenAI API returns usage info.
        # This method is for client-side estimation if needed, like for preprompt components.
        return len(self.encoding.encode(text))
