# Optimizing Multi-Skill Agent Performance

Optimizing the performance of FBA-Bench's multi-skill agents involves strategies to reduce computational cost, improve decision-making speed, and enhance overall simulation efficiency. This guide outlines key areas for performance tuning.

## 1. Skill Selection and Granularity

-   **Enable Only Necessary Skills**: Each active skill module adds overhead. Review your simulation objectives and enable only the skills (in `skill_config.yaml`) that are essential for the agent's behavior in that specific scenario.
-   **Optimize Skill Logic**: Ensure that the `execute` method within each custom skill module is as efficient as possible. Avoid redundant calculations or excessively complex loops.
-   **Skill Granularity**: Consider if a skill is too broad. Breaking down a very large skill into smaller, more focused sub-skills can sometimes improve clarity and allow for more targeted invocation, but also increases coordination overhead. Find the right balance.

## 2. LLM Call Optimization

Many skill modules will interact with Large Language Models (LLMs). LLM calls are often the most significant contributors to both cost and latency.

-   **Prompt Engineering**:
    -   **Conciseness**: Design prompts that are as concise as possible while retaining necessary context. Every token costs.
    -   **Specificity**: Provide clear instructions and define the expected output format (e.g., JSON schema) to reduce parsing errors and re-prompts.
    -   **Few-shot examples**: Use a small number of high-quality examples instead of lengthy instructions to guide the LLM's behavior.
-   **Model Selection**: Use the smallest, fastest LLM model that meets the quality requirements for a given skill's task. For example, a "reasoning" skill might need a more powerful model, while a "data extraction" skill might perform well with a smaller, faster model.
-   **LLM Caching**: Ensure LLM responses are effectively cached (see [`reproducibility/llm_cache.py`](reproducibility/llm_cache.py)). This significantly reduces redundant API calls for identical prompts.
-   **Batching**: When running distributed simulations, enable and configure LLM request batching (see [`infrastructure/llm_batcher.py`](infrastructure/llm_batcher.py)) to send multiple LLM requests in a single API call, reducing per-request overhead.

## 3. Skill Coordination Strategy

The `Skill Coordinator` (`agents/skill_coordinator.py`) can introduce overhead, especially with LLM-mediated conflict resolution.

-   **Choose Efficient Strategies**: For simple conflicts, prefer rule-based resolution strategies (e.g., `prioritize_financial_impact`) over LLM-mediated resolution, which incurs LLM call costs.
-   **Optimize Conflict Detection**: Ensure that the logic for identifying conflicts among proposed actions is performant.

## 4. Resource Management

-   **Memory Efficiency**: Implement efficient memory management within skills. Avoid storing excessively large states or histories if not necessary (see [`docs/cognitive-architecture/memory-integration.md`](docs/cognitive-architecture/memory-integration.md)).
-   **Tool Usage**: If skills use external tools, ensure these tools are optimized and their access patterns (e.g., API calls, database queries) are efficient.

## 5. Distributed Simulation Considerations

For large-scale simulations:
-   **Agent Runner Scaling**: Allocate sufficient `agent_runner` instances in your Docker Compose or Kubernetes deployment (see [`docs/infrastructure/distributed-simulation.md`](docs/infrastructure/distributed-simulation.md)).
-   **Load Balancing**: Ensure proper load balancing across multi-skill agents if they are deployed in a distributed environment, to prevent single points of bottleneck.
-   **Network Latency**: Minimize network latency between components (e.g., LLM clients, message queues) in a distributed setup.

## Configuration Examples

Adjusting `skill_config.yaml` for performance:

```yaml
# Example agents/skill_config.yaml snippet for performance
multi_skill_system:
  enabled: true
  enabled_skills:
    - supply_manager
    - marketing_manager # Only minimum necessary skills
  coordination:
    enabled: true
    conflict_resolution_strategy: "prioritize_financial_impact" # Prefer rule-based
    llm_for_resolution:
      enabled: false # Disable LLM for coordination if not strictly needed
  skill_specific_configs: # Adjust configs per skill if desired
    marketing_manager:
      llm_model: "claude-3-haiku" # Use a faster/cheaper model for this skill