import tiktoken
from typing import Dict, Any

class TokenCounter:
    def __init__(self):
        # Initialize tokenizers for common models.
        # This will be extended to support Anthropic and other models.
        self.tokenizer_cache = {
            "gpt-4": tiktoken.encoding_for_model("gpt-4"),
            "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
            # Add other models as needed, e.g., Anthropic models will require their own libraries
        }

    def count_tokens(self, text: str, model_name: str = "gpt-4") -> int:
        """
        Counts tokens in a given text using the specified model's tokenizer.
        Default to GPT-4 if model_name is not recognized or provided.
        """
        encoding = self.tokenizer_cache.get(model_name)
        if encoding is None:
            # Fallback to a default tokenizer if the specific model is not cached
            # Or raise an error if strict model adherence is required
            print(f"Warning: Tokenizer not found for model '{model_name}'. Using 'gpt-4' tokenizer.")
            encoding = self.tokenizer_cache["gpt-4"]
        
        return len(encoding.encode(text))

    def count_message_tokens(self, messages: list[Dict[str, str]], model_name: str = "gpt-4") -> int:
        """
        Counts tokens for a list of messages (e.g., for OpenAI chat completions).
        Follows OpenAI's cookbook recommendations for token counting.
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print(f"Warning: Tokenizer not found for model '{model_name}'. Using 'cl100k_base' tokenizer.")
            encoding = tiktoken.get_encoding("cl100k_base") # Universal fallback

        # From OpenAI's cookbook:
        # Every message follows 'role\ncontent\n' format, some also have 'name\n'
        # The exact count varies by model, but this is a good approximation.
        tokens_per_message = 3
        tokens_per_name = 1

        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message
            for key, value in message.items():
                total_tokens += len(encoding.encode(value))
                if key == "name":
                    total_tokens += tokens_per_name
        total_tokens += 3  # every reply is primed with assistant
        return total_tokens

    def calculate_cost(self, tokens: int, token_cost_per_1k: float) -> float:
        """Calculates the estimated cost based on tokens and cost per 1k tokens."""
        return (tokens / 1000) * token_cost_per_1k
