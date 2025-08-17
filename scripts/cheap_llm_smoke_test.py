import os
import sys
import asyncio
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so intra-repo imports work when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_interface.openrouter_client import OpenRouterClient
from llm_interface.generic_openai_client import GenericOpenAIClient
from llm_interface.schema_validator import validate_llm_response

logger = logging.getLogger("cheap_llm_smoke_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def run_test(
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    request_id: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> int:
    """
    Execute a cheap LLM smoke test against either:
      - provider=openrouter (OpenRouter chat completions)
      - provider=openai-compat (OpenAI-compatible endpoints like Together AI or local gateways)

    Returns process exit code (0 success, non-zero on failure).
    """
    client = None

    try:
        if provider == "openrouter":
            # API key precedence: explicit arg -> env OPENROUTER_API_KEY
            resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not resolved_key:
                logger.error("OPENROUTER_API_KEY environment variable is not set and no --api-key provided.")
                return 2
            # Allow overriding base URL via env if provided
            resolved_base = base_url or os.getenv("OPENROUTER_BASE_URL") or None
            client = OpenRouterClient(model_name=model, api_key=resolved_key, base_url=resolved_base)

        elif provider == "openai-compat":
            # Supports TogetherAI, local OpenAI-compatible gateways (e.g., Ollama with OpenAI API)
            # API key precedence: explicit arg -> TOGETHER_API_KEY -> OPENAI_API_KEY
            resolved_key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not resolved_key:
                logger.error("Provide --api-key or set TOGETHER_API_KEY/OPENAI_API_KEY for openai-compat provider.")
                return 2
            # Base URL precedence: explicit arg -> env OPENAI_COMPAT_BASE_URL -> Together default
            resolved_base = base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or "https://api.together.xyz/v1"
            client = GenericOpenAIClient(model_name=model, api_key=resolved_key, base_url=resolved_base)

        else:
            logger.error(f"Unsupported provider: {provider}")
            return 2

        # Prompt expects strict JSON-only output
        prompt = (
            "You are an FBA-Bench agent. Respond ONLY with JSON adhering to this schema: "
            '{"actions":[{"type":"wait_next_day"}],"reasoning":"<brief>","confidence":0.5}. '
            "Output a single JSON object with exactly these keys. No prose."
        )

        response = await client.generate_response(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            request_id=request_id or "",
        )

        content = response["choices"][0]["message"]["content"]
        print("RAW_CONTENT_START")
        print(content)
        print("RAW_CONTENT_END")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Model returned non-JSON content: {e}")
            return 3

        valid, err = validate_llm_response(parsed)
        if not valid:
            logger.error(f"Schema validation failed: {err}")
            return 4

        logger.info("Schema validation succeeded.")
        print("PARSED_JSON_START")
        print(json.dumps(parsed, indent=2))
        print("PARSED_JSON_END")

        # Print usage info if present
        usage = response.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            logger.info(f"Usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

        return 0

    except Exception as e:
        logger.exception(f"LLM call failed: {e}")
        return 1

    finally:
        # Close HTTP client if present
        try:
            if client and getattr(client, "http_client", None):
                await client.http_client.aclose()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cheap LLM smoke test (OpenRouter or OpenAI-compatible)")
    p.add_argument("--provider", choices=["openrouter", "openai-compat"], default=os.getenv("CHEAP_LLM_PROVIDER", "openrouter"))
    p.add_argument("--model", default=os.getenv("CHEAP_LLM_MODEL", "openai/gpt-4o-mini"), help="Model ID (provider-specific)")
    p.add_argument("--temperature", type=float, default=float(os.getenv("CHEAP_LLM_TEMPERATURE", "0.1")))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("CHEAP_LLM_MAX_TOKENS", "200")))
    p.add_argument("--request-id", default=os.getenv("CHEAP_LLM_REQUEST_ID", None), help="Optional request id for tracing")
    p.add_argument("--base-url", default=os.getenv("CHEAP_LLM_BASE_URL", None), help="Override API base URL")
    p.add_argument("--api-key", default=os.getenv("CHEAP_LLM_API_KEY", None), help="Override API key (takes precedence over env defaults)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(
        run_test(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_id=args.request_id,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    )
    raise SystemExit(exit_code)