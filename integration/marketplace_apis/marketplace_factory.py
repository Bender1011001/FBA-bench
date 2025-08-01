import logging
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .amazon_seller_central import AmazonSellerCentralAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketplaceFactory:
    """
    Factory pattern for different marketplace integrations.
    Handles configuration management, rate limiting, and provides a standardized interface.
    """
    _instances: Dict[str, Any] = {}
    _lock = threading.Lock()
    _rate_limiters: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        # This constructor is intentionally empty.
        # Use get_marketplace_client to obtain instances.
        pass

    @classmethod
    def get_marketplace_client(cls, platform: str, config: Dict[str, Any]) -> Any:
        """
        Retrieves or creates a marketplace API client instance.
        
        :param platform: The name of the marketplace platform (e.g., "amazon_seller_central").
        :param config: Configuration dictionary for the API client (e.g., credentials, sandbox_mode).
        :return: An instance of the requested marketplace API client.
        :raises ValueError: If the requested platform is not supported.
        """
        key = f"{platform}-{hash(json.dumps(config, sort_keys=True))}" # Unique key for client instance
        
        with cls._lock:
            if key not in cls._instances:
                logging.info(f"Creating new marketplace client for platform: {platform}")
                client = cls._create_client(platform, config)
                cls._instances[key] = client
                cls._initialize_rate_limiter(platform, config.get("rate_limit", {}))
            else:
                logging.debug(f"Returning existing marketplace client for platform: {platform}")
            return cls._instances[key]

    @classmethod
    def _create_client(cls, platform: str, config: Dict[str, Any]) -> Any:
        """Internal method to instantiate the correct client based on platform."""
        if platform == "amazon_seller_central":
            api_key = config.get("api_key", os.getenv("AMAZON_API_KEY"))
            secret_key = config.get("secret_key", os.getenv("AMAZON_SECRET_KEY"))
            sandbox_mode = config.get("sandbox_mode", True)
            if not api_key or not secret_key:
                logging.warning("Amazon API keys not provided in config or env vars. Using mock.")
            return AmazonSellerCentralAPI(api_key=api_key, secret_key=secret_key, sandbox_mode=sandbox_mode)
        # Add more marketplace clients here
        # elif platform == "etsy_api":
        #    return EtsyAPI(config)
        else:
            raise ValueError(f"Unsupported marketplace platform: {platform}")

    @classmethod
    def _initialize_rate_limiter(cls, platform: str, rate_limit_config: Dict[str, Any]):
        """Initializes a basic rate limiter for a given platform."""
        max_calls = rate_limit_config.get("max_calls", 10)  # e.g., 10 calls
        period = rate_limit_config.get("period", 1)  # e.g., per 1 second
        
        cls._rate_limiters[platform] = {
            "max_calls": max_calls,
            "period": period,
            "calls": deque(),
            "lock": threading.Lock()
        }
        logging.info(f"Rate limiter initialized for {platform}: {max_calls} calls per {period}s.")

    @classmethod
    async def enforce_rate_limit(cls, platform: str):
        """
        Enforces rate limits before allowing an API call.
        This method should be called before attempting an API operation.
        """
        limiter = cls._rate_limiters.get(platform)
        if not limiter:
            logging.debug(f"No rate limiter configured for {platform}. Skipping enforcement.")
            return

        with limiter["lock"]:
            now = time.monotonic()
            # Remove calls older than the period
            while limiter["calls"] and limiter["calls"][0] <= now - limiter["period"]:
                limiter["calls"].popleft()

            if len(limiter["calls"]) >= limiter["max_calls"]:
                wait_time = limiter["period"] - (now - limiter["calls"][0])
                logging.warning(f"Rate limit hit for {platform}. Waiting for {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)
                # After waiting, re-check and potentially wait again if other threads
                # made calls during the wait. This is a very basic implementation.
                while limiter["calls"] and limiter["calls"][0] <= time.monotonic() - limiter["period"]:
                    limiter["calls"].popleft()
                if len(limiter["calls"]) >= limiter["max_calls"]:
                    # This could happen if, due to concurrency, by the time we wake up,
                    # the window has refilled but immediately used by other threads.
                    # For simplicity, we assume one sufficient wait.
                    # A more robust solution might involve token buckets or semaphores.
                    pass 
            limiter["calls"].append(time.monotonic())
            logging.debug(f"Rate limit check passed for {platform}. Calls in window: {len(limiter['calls'])}")

# Example usage (for internal testing)
import json # Ensure json is imported at the top as well

async def _main():
    # Example for Amazon Seller Central
    amazon_config = {
        "api_key": "YOUR_AMAZON_API_KEY", # In a real app, load from secure config/env
        "secret_key": "YOUR_AMAZON_SECRET_KEY",
        "sandbox_mode": True,
        "rate_limit": {"max_calls": 3, "period": 5} # 3 calls per 5 seconds
    }

    amazon_client = MarketplaceFactory.get_marketplace_client("amazon_seller_central", amazon_config)
    print("Amazon Client obtained.")

    # Test rate limiting
    logging.info("Testing Amazon rate limit...")
    for i in range(5):
        try:
            await MarketplaceFactory.enforce_rate_limit("amazon_seller_central")
            print(f"Call {i+1} to Amazon API (simulated)")
            # Simulate an API call
            _ = await amazon_client.get_inventory_summary()
            await asyncio.sleep(0.5) # Simulate work
        except Exception as e:
            logging.error(f"Error during API call: {e}")

    # Example for a hypothetical Etsy client
    # etsy_config = {
    #     "api_key": "YOUR_ETSY_API_KEY",
    #     "rate_limit": {"max_calls": 5, "period": 1}
    # }
    # try:
    #     etsy_client = MarketplaceFactory.get_marketplace_client("etsy_api", etsy_config)
    #     print("Etsy Client obtained.")
    # except ValueError as e:
    #     print(e)
    #
    # logging.info("Testing Etsy rate limit (should fail as client is not implemented)...")
    # try:
    #     await MarketplaceFactory.enforce_rate_limit("etsy_api")
    # except ValueError as e:
    #     print(f"Expected error: {e}")


if __name__ == "__main__":
    import os
    from collections import deque # Re-import deque if this file is run standalone
    import asyncio
    asyncio.run(_main())