# LLM Batching and Cost Optimization

Large Language Model (LLM) interactions are often the most significant cost and latency drivers in FBA-Bench simulations. This guide details strategies and configurations for optimizing LLM usage, particularly through request batching and careful model selection.

## LLM Request Batching

LLM batching is a technique where multiple individual LLM prompts are grouped together and sent to the LLM provider in a single API call. This can significantly reduce per-request overhead (like network latency and API call fixed costs) and improve overall throughput.

### How it Works

The `LLMBatcher` (implemented in [`infrastructure/llm_batcher.py`](infrastructure/llm_batcher.py)) intercepts outgoing LLM requests from agents and other components. Instead of sending each request immediately, it queues them up for a short period. Once the queue reaches a configurable batch size or a timeout occurs, all queued requests are sent as a single batched request to the LLM provider. The responses are then fanned out to their respective callers.

### Configuration

LLM batching is typically configured in `infrastructure/scalability_config.yaml` or directly when initializing LLM clients that support a batching mechanism.

```yaml
# Example infrastructure/scalability_config.yaml snippet
llm_orchestration:
  batching:
    enabled: true
    batch_size: 50 # Number of LLM requests to group into a single batch
    batch_interval_seconds: 0.1 # Max time (in seconds) to wait before sending a batch
    max_concurrent_batches: 5 # Max number of batches to send concurrently
  caching:
    enabled: true # Ensure LLM caching is enabled for further optimization
```

### Important Considerations:
-   **LLM Provider Support**: Not all LLM providers natively support batching of arbitrary requests. FBA-Bench's internal LLM clients (e.g., [`llm_interface/openrouter_client.py`](llm_interface/openrouter_client.py)) are designed to work with providers that offer this capability.
-   **Latency Trade-off**: While batching improves throughput and reduces cost, it can slightly increase the latency for individual requests as they wait in the queue. Tune `batch_interval_seconds` to balance this trade-off.
-   **Distributed Environments**: Batching is particularly effective in distributed simulation setups where many agents might be making LLM calls simultaneously.

## Cost Mitigation Strategies

Beyond batching, several strategies can help reduce LLM-related expenses:

### 1. Model Selection
-   **Tiered Models**: Utilize different LLM models based on the complexity and criticality of the task. For simpler tasks (e.g., summarizing short observations), use smaller, faster, and cheaper models (e.g., GPT-3.5 variants, Claude Haiku). Reserve more powerful, expensive models (e.g., GPT-4, Claude Opus) for complex reasoning, planning, or reflection tasks.
-   **API Providers**: Compare pricing across different LLM API providers.

### 2. Prompt Engineering
-   **Token Efficiency**: Craft concise and precise prompts. Avoid verbose preambles or unnecessary context. Every token counts towards billing.
-   **Tool Use**: Where possible, let agents use deterministic tools (Python functions, rule engines) for tasks rather than relying on LLMs, especially for calculations or data lookups.
-   **Instruction vs. Examples**: Balance detailed instructions with few-shot examples. Sometimes a well-chosen example can reduce prompt length more effectively than extensive textual instructions.

### 3. Caching
-   **LLM Caching (`reproducibility/llm_cache.py`)**: Implement and utilize LLM response caching aggressively. If an identical prompt has been sent before (and the cache is valid), the response can be retrieved from the cache instead of incurring a new LLM call. This is one of the most effective cost-saving measures.

### 4. Response Parsing and Validation
-   **Structured Outputs**: Request LLMs to provide structured outputs (e.g., JSON). This reduces the need for re-prompts due to parsing failures, saving tokens and time. Ensure robust schema validation (see [`llm_interface/schema_validator.py`](llm_interface/schema_validator.py)).

## Example Configuration

```yaml
# Example agent/cognitive_config.yaml with model selection per task
cognitive_system:
  hierarchical_planning:
    llm_model: "gpt-4o" # More powerful for planning
  reflection:
    llm_model: "claude-3-opus" # High-quality for insights
  memory:
    llm_model: "gpt-3.5-turbo" # Cheaper for memory validation/summarization

# Example skill_config.yaml snippet for skill-specific model usage
multi_skill_system:
  skills:
    marketing_manager:
      llm_model: "claude-3-haiku" # Fast/cheap for marketing decisions
    financial_analyst:
      llm_model: "gpt-4o" # Powerful for financial analysis
```

For more on infrastructure setup and deployment, refer to [`Distributed Simulation`](distributed-simulation.md) and [`Deployment Guide`](deployment-guide.md).