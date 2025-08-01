# Using FBA-Bench for Agent Benchmarking

FBA-Bench is explicitly designed as a benchmarking platform for agentic AI systems. This guide provides best practices for setting up, executing, and reporting scientific benchmarks to compare different agent architectures, LLM models, or experimental configurations.

## 1. Evaluation Methodologies and Metrics

### Choose Relevant Metrics
FBA-Bench captures a wide array of metrics within the `metrics/` directory. Select the ones most relevant to your benchmarking objective.

-   **Business Metrics**: (`metrics/finance_metrics.py`, `metrics/marketing_metrics.py`, `metrics/operations_metrics.py`)
    -   Net Profit, Revenue, Market Share, Customer Satisfaction, Inventory Turnover, etc.
-   **Cognitive Metrics**: (`metrics/cognitive_metrics.py`)
    -   LLM Token Usage, API Call Latency, Planning Coherence Score, Reflection Quality Score, Memory Consistency Score.
-   **Cost Metrics**: (`metrics/cost_metrics.py`)
    -   Total LLM Cost, Compute Cost, Operational Cost.
-   **Robustness/Safety Metrics**: (`metrics/stress_metrics.py`, `metrics/trust_metrics.py`, `metrics/adversarial_metrics.py`)
    -   Constraint Violation Count, Recovery Rate, Success Rate under Adversarial Conditions.

### Define Success Criteria
For each benchmark, clearly define what constitutes "success" from a quantitative perspective (e.g., "Agent X achieves >$50,000 net profit in Scenario Y", "Agent Z has <5% LLM cost compared to Agent W").

### Baselining
Always include a baseline agent (e.g., a simple heuristic agent, or a previous version of your agent) for comparative analysis to understand the meaningfulness of improvements.

## 2. Reproducibility and Scientific Rigor

Reproducibility is paramount for valid scientific benchmarking.

-   **Fixed Seed**: Always set and log the simulation seed (`reproducibility/sim_seed.py`) for all runs to ensure deterministic marketplace events and initial random states.
-   **LLM Caching**: Enable and use LLM caching (`reproducibility/llm_cache.py`) to eliminate non-determinism from LLM API calls. This ensures identical prompts yield identical responses.
-   **Golden Masters**: Use the [`Golden Master`](../scenarios/scenario-validation.md) feature to capture and validate expected traces for reference scenarios. Any deviation signals a change in behavior, intended or otherwise.
-   **Version Control**: Precisely version control your entire experimental setup:
    -   Agent code and configuration (e.g., `agents/`)
    -   Scenario definitions and any custom events (e.g., `scenarios/`)
    -   FBA-Bench platform version (via Git commit hash)
    -   `requirements.txt` for all dependencies.
-   **Environment Consistency**: Use containerization (Docker, Kubernetes) to ensure a consistent execution environment.
-   **Detailed Logging**: Enable comprehensive tracing (`observability/` config) to capture every step, LLM call, and tool use for post-hoc debugging if results are unexpected.

## 3. Comparative Analysis Guidelines

When comparing different agent versions or models:

-   **Aligned Configurations**: Ensure all non-tested parameters (scenario, other cognitive settings, etc.) are identical across different agents.
-   **Multiple Replicates**: For stochastic components, run each agent/configuration multiple times (e.g., 5-10 replicates) with different seeds to account for statistical variance and report average performance with confidence intervals.
-   **Statistical Tests**: Apply appropriate statistical tests (e.g., t-tests, ANOVA) to determine if observed differences in performance are statistically significant.
-   **Visualization**: Use clear charts and graphs to illustrate performance differences across metrics. The FBA-Bench frontend's `Comparison Tool` (`frontend/src/components/ComparisonTool.tsx`) can assist.

## 4. Reporting Best Practices

A good benchmark report should be transparent and comprehensive:

-   **Executive Summary**: Briefly state the objective, methodology, key findings, and conclusions.
-   **Methodology**:
    -   Detailed description of agents (architecture, LLMs used, key configs).
    -   Full description of scenarios (type, duration, key events, goals, tier).
    -   LLM details (model name, version, temperature, prompt engineering strategies).
    -   Compute resources used.
    -   Number of replicates and statistical analysis methods.
    -   Reproducibility details (seed, caching status, git commit hash).
-   **Results**:
    -   Present quantitative results clearly using tables and charts.
    -   Report mean and standard deviation for metrics.
    -   Highlight statistically significant findings.
    -   Include qualitative observations from trace analysis.
-   **Discussion**:
    -   Interpret the results in the context of your hypotheses.
    -   Discuss limitations of the study.
    -   Propose future work or improvements.
-   **Appendices**: Include full configuration files, example traces, and raw data if feasible.

By following this guide, your research with FBA-Bench will contribute to a more rigorous and reliable understanding of agentic AI capabilities.