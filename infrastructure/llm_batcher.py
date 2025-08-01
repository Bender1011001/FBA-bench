import asyncio
import time
import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional, Tuple

# Assuming BaseEvent and other necessary types/classes are available in the FBA-Bench context
# For simplicity, we'll define a basic structure for LLM request/response here.
# In a real scenario, these would likely integrate with an existing LLM client.

logger = logging.getLogger(__name__)

class LLMRequest:
    """Represents a single LLM request."""
    def __init__(self, request_id: str, prompt: str, model: str, callback: Callable):
        self.request_id = request_id
        self.prompt = prompt
        self.model = model
        self.callback = callback
        self.timestamp = time.time()
        self.batch_id: Optional[str] = None
        self.status: str = "pending"
        self.response: Any = None
        self.error: Optional[Exception] = None

class LLMBatcher:
    """
    Manages LLM request batching for cost optimization and throughput.

    - Request aggregation: Collects multiple LLM requests for batch processing.
    - Intelligent batching: Groups similar requests to maximize API efficiency.
    - Adaptive timing: Balances latency vs. throughput based on system load.
    - Cost optimization: Minimizes token usage through request deduplication.
    """

    def __init__(self):
        self._pending_requests: Dict[str, LLMRequest] = {}
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        
        self.max_batch_size: int = 10
        self.batch_timeout_ms: int = 500
        self.similarity_threshold: float = 0.8 # For future advanced deduplication/grouping
        
        self.stats = {
            "total_requests_received": 0,
            "total_requests_batched": 0,
            "total_batches_processed": 0,
            "total_tokens_estimated": 0,
            "total_api_cost_estimated": 0.0,
            "requests_deduplicated": 0,
            "avg_batch_size": 0.0,
            "max_batch_latency_ms": 0.0,
        }

    async def start(self):
        """Starts the batcher's background processing task."""
        if self._running:
            logger.warning("LLMBatcher already running.")
            return
        self._running = True
        self._processing_task = asyncio.create_task(self._batching_loop())
        logger.info("LLMBatcher started.")

    async def stop(self):
        """Stops the batcher's background processing task."""
        if not self._running:
            return
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("LLMBatcher stopped.")

    def add_request(self, request_id: str, prompt: str, model: str, callback: Callable):
        """
        Queues an LLM request for batch processing.
        
        Args:
            request_id: Unique identifier for the request.
            prompt: The LLM prompt string.
            model: The LLM model to use (e.g., "gpt-4", "claude-sonnet").
            callback: An async callable to invoke with the response (or error).
        """
        self.stats["total_requests_received"] += 1
        new_request = LLMRequest(request_id, prompt, model, callback)
        if request_id in self._pending_requests:
            logger.warning(f"Request ID {request_id} already exists, overwriting.")
        self._pending_requests[request_id] = new_request
        logger.debug(f"Added request {request_id} for model {model}.")
    
    def set_batch_parameters(self, max_size: int = None, timeout_ms: int = None, similarity_threshold: float = None):
        """
        Configures batching parameters.

        Args:
            max_size: Maximum number of requests per batch.
            timeout_ms: Maximum time (in milliseconds) to wait before sending a batch.
            similarity_threshold: Threshold for grouping similar requests (0.0-1.0).
        """
        if max_size is not None:
            self.max_batch_size = max_size
        if timeout_ms is not None:
            self.batch_timeout_ms = timeout_ms
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        logger.info(f"LLMBatcher parameters updated: max_size={self.max_batch_size}, timeout={self.batch_timeout_ms}ms, similarity_threshold={self.similarity_threshold}")


    async def _batching_loop(self):
        """Continuously aggregates and processes batches."""
        logger.info("LLMBatching loop started.")
        while self._running:
            try:
                await self._aggregate_and_process()
            except Exception as e:
                logger.error(f"Error in batching loop: {e}", exc_info=True)
            await asyncio.sleep(self.batch_timeout_ms / 1000.0 / 2) # Check more frequently than timeout

        logger.info("LLMBatching loop stopped.")

    async def _aggregate_and_process(self):
        """Aggregates pending requests into batches and processes them."""
        if not self._pending_requests:
            return

        current_time = time.time()
        requests_to_batch = []
        requests_to_remove_from_pending = []

        # Collect requests that are ready to be batched (either by timeout or if we hit max size)
        for req_id, req in list(self._pending_requests.items()): # copy to allow modification during iteration
            # Always ensure we have enough space in the batch
            if len(requests_to_batch) < self.max_batch_size and (current_time - req.timestamp) * 1000 >= self.batch_timeout_ms:
                requests_to_batch.append(req)
                requests_to_remove_from_pending.append(req_id)
            elif len(requests_to_batch) >= self.max_batch_size:
                # If we've hit max_batch_size, process what we have and then continue in next loop iteration
                break

        if requests_to_batch:
            # Remove batched requests from pending
            for req_id in requests_to_remove_from_pending:
                self._pending_requests.pop(req_id, None)

            # Process the batch
            processed_batch = self.optimize_batch_composition(requests_to_batch)
            self._update_stats_from_batch(processed_batch)
            await self.process_batch(processed_batch)
            
            # Update average batch size
            if self.stats["total_batches_processed"] > 0:
                self.stats["avg_batch_size"] = self.stats["total_requests_batched"] / self.stats["total_batches_processed"]
            
            # Update max batch latency
            batch_latency = (time.time() - min(r.timestamp for r in processed_batch if r.timestamp is not None)) * 1000
            self.stats["max_batch_latency_ms"] = max(self.stats["max_batch_latency_ms"], batch_latency)


    def optimize_batch_composition(self, requests: List[LLMRequest]) -> Dict[str, List[LLMRequest]]:
        """
        Groups similar requests for a single LLM API call.
        Prioritizes grouping by model, then by potential for deduplication or similar context.

        Args:
            requests: A list of LLMRequest objects.

        Returns:
            A dictionary where keys are a composite identifier (model_deduplicated_prompt_hash)
            and values are lists of LLMRequest objects forming a batch.
        """
        optimized_batches: Dict[str, List[LLMRequest]] = defaultdict(list)
        deduplicated_prompts: Dict[Tuple[str, str], str] = {} # (model, prompt) -> hash of prompt

        for req in requests:
            # Simple deduplication: store a hash of the prompt for a given model
            # Key for deduplication is (model, prompt content)
            prompt_key = (req.model, req.prompt)
            if prompt_key not in deduplicated_prompts:
                # Assign a unique hash for prompts to serve as a batch key
                prompt_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.prompt + req.model))
                deduplicated_prompts[prompt_key] = prompt_hash
            else:
                self.stats["requests_deduplicated"] += 1
                logger.debug(f"Deduplicated request {req.request_id} (prompt already seen for model {req.model}).")

            # Create a batch key based on model and the deduplicated prompt (or unique prompt hash)
            batch_key = f"{req.model}_{deduplicated_prompts[prompt_key]}"
            optimized_batches[batch_key].append(req)
            req.batch_id = batch_key
            logger.debug(f"Assigned request {req.request_id} to batch {batch_key}.")
            
        return optimized_batches

    async def process_batch(self, batched_requests: Dict[str, List[LLMRequest]]):
        """
        Sends batched requests to the LLM API and handles responses.
        This method would interact with the actual LLM API client.
        For demonstration, it simulates an async API call.
        """
        if not batched_requests:
            return

        logger.info(f"Processing {len(batched_requests)} optimized batches.")

        processor_tasks = []
        for batch_key, requests_in_batch in batched_requests.items():
            # Simulate sending a single request per unique prompt/model combination
            # All requests in 'requests_in_batch' share the same prompt and model
            primary_request_for_batch = requests_in_batch[0] 
            
            # In a real scenario, this would use an actual LLM client, e.g., OpenAI, Anthropic
            task = asyncio.create_task(
                self._execute_llm_api_call(
                    primary_request_for_batch.prompt,
                    primary_request_for_batch.model,
                    requests_in_batch # Pass all original requests to distribute response
                )
            )
            processor_tasks.append(task)
            
        if processor_tasks:
            await asyncio.gather(*processor_tasks, return_exceptions=True)
            logger.info(f"Completed processing {len(batched_requests)} optimized batches.")
        else:
            logger.warning("No processor tasks were created for the batches.")


    async def _execute_llm_api_call(self, prompt: str, model: str, original_requests: List[LLMRequest]):
        """Simulates an LLM API call and distributes the result to original request callbacks."""
        try:
            # Simulate API call latency and token usage
            await asyncio.sleep(0.1 + len(prompt) / 10000.0) # Longer prompts take more time
            
            simulated_response = f"Simulated response to '{prompt[:50]}...' from model {model}"
            
            estimated_tokens = len(prompt) / 4 + 50 # rough estimate
            estimated_cost = (estimated_tokens / 1000) * 0.002 # Example cost per 1k tokens

            self.stats["total_tokens_estimated"] += estimated_tokens
            self.stats["total_api_cost_estimated"] += estimated_cost

            for req in original_requests:
                req.status = "completed"
                req.response = simulated_response
                req.callback(req.request_id, req.response, None) # Invoke original callback

            logger.info(f"Simulated LLM call for model {model} (Prompt: '{prompt[:30].strip()}...'). Estimated tokens: {estimated_tokens:.2f}")

        except Exception as e:
            logger.error(f"Error during simulated LLM API call for model {model}: {e}", exc_info=True)
            for req in original_requests:
                req.status = "failed"
                req.error = e
                req.callback(req.request_id, None, e) # Invoke original callback with error


    def estimate_batch_cost(self, batch_of_requests: Dict[str, List[LLMRequest]]) -> Tuple[int, float]:
        """
        Calculates estimated token usage and API costs for a given batch.
        Assumes basic token estimation and a fixed cost per 1k tokens.

        Args:
            batch_of_requests: A dictionary of batched requests as returned by optimize_batch_composition.

        Returns:
            A tuple of (estimated_tokens, estimated_cost).
        """
        total_estimated_tokens = 0
        total_estimated_cost = 0.0

        for batch_key, requests_in_batch in batch_of_requests.items():
            # For cost estimation, we only count the unique prompt within each optimized batch
            if requests_in_batch:
                unique_prompt = requests_in_batch[0].prompt
                # A very crude token estimation (e.g., 1 token per 4 characters + a fixed response overhead)
                estimated_prompt_tokens = len(unique_prompt) / 4 
                
                # Assume a fixed response token count for simplicity
                estimated_response_tokens = 50 
                
                batch_tokens = estimated_prompt_tokens + estimated_response_tokens
                total_estimated_tokens += batch_tokens
                
                # Example cost model: $0.002 per 1k tokens (adjust based on actual LLM pricing)
                cost_per_1k_tokens = 0.002 
                total_estimated_cost += (batch_tokens / 1000) * cost_per_1k_tokens
        
        logger.debug(f"Estimated cost for batch: {total_estimated_cost:.6f} USD, {total_estimated_tokens:.2f} tokens.")
        return int(total_estimated_tokens), total_estimated_cost

    def _update_stats_from_batch(self, batched_requests: Dict[str, List[LLMRequest]]):
        """Internal helper to update statistics based on processed batches."""
        self.stats["total_batches_processed"] += len(batched_requests)
        for _, requests_in_batch in batched_requests.items():
            self.stats["total_requests_batched"] += len(requests_in_batch)
            # Cost and token estimation will be handled by the _execute_llm_api_call for now
            # as it simulates the actual API interaction.
            # If we wanted to pre-estimate, we'd call estimate_batch_cost here.