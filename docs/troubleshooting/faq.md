# Frequently Asked Questions (FAQ)

This section addresses common questions about FBA-Bench's enhanced features, migration from older versions, and general operational aspects.

## 1. About New Features

-   **Q: What are the main improvements in this enhanced version of FBA-Bench?**
    -   A: The enhanced FBA-Bench introduces advanced cognitive architectures (hierarchical planning, reflection, validated memory), multi-skill agent capabilities, infrastructure scalability (distributed simulation, LLM batching), rich scenario diversity, enhanced tool interfaces and observability, agent learning features, and real-world integration pathways. See the [`Master Documentation Overview`](../README.md) for a summary.

-   **Q: How do I enable the new cognitive features like planning and reflection?**
    -   A: These features are generally enabled by default for `AdvancedAgent` types. You can configure them via `cognitive_config.yaml` to adjust parameters like planning depth, reflection frequency, or memory validation. See [`Cognitive System Configuration`](../configuration/cognitive-config.md).

-   **Q: Can I run multiple agents simultaneously?**
    -   A: Yes, the enhanced FBA-Bench supports distributed simulation, allowing you to run many agents concurrently. Refer to the [`Distributed Simulation`](../infrastructure/distributed-simulation.md) guide for setup instructions using Docker Compose or Kubernetes.

-   **Q: How can I reduce the cost of LLM API calls?**
    -   A: Utilize LLM batching (configured in `infrastructure-config.yaml`), enable LLM caching (`llm_cache.py`), choose smaller/cheaper LLM models for less critical tasks, and optimize your prompts to reduce token usage. See [`LLM Batching and Cost Optimization`](../infrastructure/performance-optimization.md) for detailed strategies.

-   **Q: What kind of scenarios can I simulate now?**
    -   A: Beyond basic simulations, you can run comprehensive business scenarios (e.g., international expansion, supply chain crisis), multi-agent cooperation and competition, and dynamically generated scenarios. FBA-Bench also supports curriculum-based evaluations across T0-T3 difficulty tiers. See [`Scenario System Overview`](../scenarios/scenario-system.md).

-   **Q: How do I get detailed insights into my agent's thought process?**
    -   A: FBA-Bench captures extensive simulation traces. Use the `TraceAnalyzer` API or the frontend's `TraceViewer` to delve into agent decisions, LLM interactions, and tool usage. Ensure tracing is enabled in `observability_config.yaml`. See [`Advanced Trace Analysis`](../observability/trace-analysis.md).

## 2. Migration from Older Versions (Conceptual)

-   **Q: I have existing agents and scenarios from an older FBA-Bench version. Are they compatible?**
    -   A: While general concepts remain, the enhanced version introduces new configuration schemas and API interfaces for cognitive, multi-skill, and infrastructure components. Existing agent implementations and scenario YAMLs may require updates to align with the new structure. Refer to dedicated migration guides for a smooth transition. (Note: A formal migration guide is currently outside the scope of this document but would detail specific changes).
-   **Q: Will my old `requirements.txt` still work?**
    -   A: The enhanced version likely has updated or new dependencies. It's recommended to use the provided `requirements.txt` (and `requirements-dev.txt` if available) from the enhanced repository.

## 3. Performance and Scalability Questions

-   **Q: My simulation is running slowly even with distributed setup.**
    -   A: Check `agent_runner_concurrency` in `infrastructure_config.yaml`. Ensure sufficient compute resources (CPU/RAM) are allocated to your Docker containers or Kubernetes pods. Profile your agent's LLM usage. Extensive LLM calls, especially with large contexts or expensive models, are often the bottleneck. Look into `max_concurrent_batches` in `llm_orchestration` settings. See [`Optimizing Multi-Skill Agent Performance`](../multi-skill-agents/performance-optimization.md) and [`LLM Batching and Cost Optimization`](../infrastructure/performance-optimization.md).
-   **Q: How do I monitor the health of my distributed FBA-Bench deployment?**
    -   A: Use the FBA-Bench frontend dashboards (`frontend/`) which connect to the backend API server. Configure `performance_monitoring` in `observability_config.yaml` to export metrics. For critical production deployments, integrate with external tools like Prometheus/Grafana or Datadog.

## 4. Safety and Security Considerations

-   **Q: How safe is it to connect agents to real marketplaces?**
    -   A: FBA-Bench prioritizes safety. The `RealWorldAdapter` enforces configurable `safety_constraints` (e.g., max price change, manual approval thresholds). Always enable `dry_run_mode` during development and testing. **Never deploy to live without thorough testing in a sandboxed environment first.** See [`Real-World Integration`](../integration/real-world-integration.md) and [`Safety Constraints`](../integration/safety-constraints.md).
-   **Q: Can I prevent an agent from making extreme decisions?**
    -   A: Yes, you can define `safety_constraints` in `integration_config.yaml` for actions interacting with the real world. For internal agent decisions, refine your agent's prompts and cognitive controls to guide its behavior and implement internal validation mechanisms within custom skills.

For more in-depth solutions, refer to the [`Common Issues and Solutions`](common-issues.md) guide.