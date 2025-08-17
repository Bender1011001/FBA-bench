import os
import httpx
import tiktoken
import logging
from typing import Dict, Any, Optional

from llm_interface.contract import BaseLLMClient, LLMClientError

logger = logging.getLogger(__name__)

class GenericOpenAIClient(BaseLLMClient):
    """
    Generic OpenAI-compatible LLM client (e.g., Together AI, local OpenAI-compatible gateways).

    Usage:
      - Provide a base_url that implements the OpenAI Chat Completions API:
          POST {base_url}/chat/completions
      - Provide an API key via the constructor or environment variable(s).

    Env overrides:
      - OPENAI_COMPAT_SITE_URL: Optional HTTP-Referer header for attribution.
      - OPENAI_COMPAT_APP_TITLE: Optional X-Title header for attribution.
    """

    def __init__(self, model_name: str, api_key: Optional[str], base_url: Optional[str]):
        super().__init__(model_name, api_key, base_url)
        if not base_url:
            raise ValueError("GenericOpenAIClient requires a base_url for the OpenAI-compatible endpoint.")
        if not api_key:
            raise ValueError("GenericOpenAIClient requires an API key. Pass api_key or set a provider-specific env var.")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http_client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

        # Initialize tiktoken for token counting (best-effort)
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            logger.warning(
                f"Could not find tiktoken encoding for model '{self.model_name}'. "
                "Using 'cl100k_base' as fallback. Token counts might be inaccurate."
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a structured response from an OpenAI-compatible API.

        Args:
            prompt: The formatted prompt string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate in the completion.
            response_format: Optional OpenAI JSON schema, e.g., {"type": "json_object"}.
            **kwargs: Additional parameters (e.g., top_p, frequency_penalty, presence_penalty).

        Returns:
            The raw OpenAI-compatible response dict.
        """
        referer = os.getenv("OPENAI_COMPAT_SITE_URL", "https://fba-bench.com")
        app_title = os.getenv("OPENAI_COMPAT_APP_TITLE", "FBA-Bench")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": referer,
            "X-Title": app_title,
            "Content-Type": "application/json",
        }
        # Move request_id from payload kwargs to header to avoid OpenAI 400 errors
        req_id = kwargs.pop("request_id", None)
        if req_id:
            headers["X-Request-Id"] = req_id

        messages = [{"role": "user", "content": prompt}]
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        # Merge any extra kwargs (e.g., top_p)
        payload.update(kwargs)

        try:
            logger.debug(f"Calling OpenAI-compatible API at {self.base_url} for model {self.model_name} with payload: {payload}")
            response = await self.http_client.post(
                "/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Received response: {response_data}")

            # Validate minimal structure
            if not response_data.get("choices") or not isinstance(response_data["choices"], list):
                raise LLMClientError(f"Response missing 'choices' list: {response_data}")
            first_choice = response_data["choices"][0]
            if not first_choice.get("message") or not first_choice["message"].get("content"):
                raise LLMClientError(f"Response missing message content in choices: {response_data}")

            return response_data

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from OpenAI-compatible API: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(
                f"OpenAI-compatible API returned an HTTP error: {e.response.status_code} - {e.response.text}",
                original_exception=e,
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            logger.error(f"Network error connecting to OpenAI-compatible API: {e}")
            raise LLMClientError(
                f"Network error connecting to OpenAI-compatible API: {e}",
                original_exception=e
            )
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI-compatible API call: {e}", exc_info=True)
            raise LLMClientError(
                f"Unexpected error during OpenAI-compatible API call: {e}",
                original_exception=e
            )

    async def get_token_count(self, text: str) -> int:
        """Best-effort token estimate using tiktoken."""
        return len(self.encoding.encode(text))