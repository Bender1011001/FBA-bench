# Common Issues and Solutions

This document provides a guide to troubleshooting common problems encountered while using FBA-Bench, along with practical solutions.

## 1. Installation and Setup Issues

### Issue: `ModuleNotFoundError` or `ImportError`
-   **Problem**: Python cannot find FBA-Bench modules or dependencies.
-   **Solution**:
    1.  Ensure you are in the correct virtual environment. Activate it using `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows).
    2.  Install all required dependencies: `pip install -r requirements.txt`.
    3.  Verify your Python path. If you moved FBA-Bench directories, Python might not find modules. Run `python -c "import sys; print(sys.path)"` to check.

### Issue: Docker Compose fails to start services
-   **Problem**: `docker-compose up` command fails with errors.
-   **Solution**:
    1.  Ensure Docker Desktop (or Docker service if on Linux server) is running.
    2.  Check for port conflicts: Another application might be using a port FBA-Bench needs (e.g., 6379 for Redis, 8000 for API server).
    3.  Run `docker-compose build` separately first to identify build errors in images.
    4.  Review network configuration in `docker-compose.yml` if containers cannot communicate.

## 2. LLM Interaction Problems

### Issue: `AuthenticationError` or `Invalid API Key`
-   **Problem**: LLM API calls fail due to incorrect or missing API keys.
-   **Solution**:
    1.  Ensure your LLM provider API key (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) is correctly set as an environment variable in your shell session or in your operating system's environment variables.
    2.  If running with Docker, ensure the environment variables are correctly passed to the container (e.g., via `env_file` or `environment` in `docker-compose.yml`).
    3.  Verify the API key is active and has necessary permissions with your LLM provider.

### Issue: `RateLimitExceededError` (429 Too Many Requests)
-   **Problem**: You are sending too many LLM requests too quickly, exceeding the API provider's rate limits.
-   **Solution**:
    1.  Enable and configure LLM batching (`llm_orchestration.batching.enabled: true` in `infrastructure_config.yaml`) to reduce the number of discrete API calls.
    2.  Increase `batch_interval_seconds` to allow more time for requests to accumulate in a batch.
    3.  Reduce the number of concurrent agents making LLM calls.
    4.  If persistent, contact your LLM provider to request a rate limit increase.

### Issue: `ContextWindowExceededError` or `MaxTokensExceeded`
-   **Problem**: The LLM prompt (or the combined prompt + response) is too long for the selected LLM model's context window.
-   **Solution**:
    1.  Review your agent's prompt engineering: make prompts more concise and remove redundant information.
    2.  Adjust `max_reflection_tokens` in `cognitive_config.yaml` to reduce the size of reflection outputs.
    3.  Refine your memory management `forgetting_strategy` to keep less relevant data out of the prompt context.
    4.  Consider using a different, larger context window LLM model for tasks requiring extensive context (e.g., Claude 3 Opus, GPT-4 Turbo).

### Issue: LLM provides malformed JSON or unexpected responses
-   **Problem**: The LLM's output does not conform to the expected structured format (e.g., JSON schema), causing parsing errors.
-   **Solution**:
    1.  **Refine Prompt Engineering**: Make your prompts more explicit about the required output format. Provide clear examples if possible.
    2.  **Increase Temperature (Slightly)**: Sometimes, a very low `temperature` can make LLMs "stuck" and produce malformed output. A slight increase (e.g., to 0.5-0.7) might help.
    3.  **Implement Retry Logic**: FBA-Bench has built-in retry mechanisms, but ensure they are configured to attempt re-prompts on parsing failures.
    4.  **Schema Validation**: Leverage FBA-Bench's `SchemaValidator` (`llm_interface/schema_validator.py`) to validate LLM output and provide specific feedback on parsing errors.

## 3. Simulation Runtime Issues

### Issue: Simulation is too slow
-   **Problem**: Simulations take a long time to complete.
-   **Solution**:
    1.  **Distributed Simulation**: Enable and configure [`Distributed Simulation`](../infrastructure/distributed-simulation.md) with multiple `agent_runner` instances.
    2.  **LLM Optimization**: Implement LLM batching and effective caching (`llm_orchestration` in `infrastructure_config.yaml`). Choose cheaper/faster models where possible.
    3.  **Fast-Forwarding**: Use `FastForwardEngine` if your scenario has periods that don't require granular simulation (e.g., periods of no agent activity).
    4.  **Agent Complexity**: Reduce cognitive processing `depth` or reflection `frequency` if agents are over-thinking.
    5.  **Hardware**: Ensure your host machine or cloud instances have sufficient CPU, memory, and network bandwidth.

### Issue: Agent behavior is irrational or inconsistent
-   **Problem**: The agent's actions don't align with expectations or change unpredictably.
-   **Solution**:
    1.  **Trace Analysis**: Use [`Advanced Trace Analysis`](../observability/trace-analysis.md) to inspect the agent's thought process, observations, and LLM calls leading up to the unexpected action.
    2.  **Memory Consistency**: Ensure `memory.validation` and `memory.consistency_checks` are enabled in `cognitive_config.yaml`.
    3.  **Skill Coordination**: If using multi-skill agents, verify your `conflict_resolution_strategy` in `skill_config.yaml` is correctly handling overlapping advice.
    4.  **Prompt Quality**: Improve the clarity and specificity of prompts provided to LLMs, especially for planning and decision-making.
    5.  **Insufficient Context**: Ensure the agent is receiving all necessary information (via its observation space and memory) to make informed decisions.

## 4. Data and Reproducibility Issues

### Issue: Simulation results are not reproducible
-   **Problem**: Running the same simulation twice with identical inputs yields different results.
-   **Solution**:
    1.  **LLM Caching**: Ensure `llm_orchestration.caching.enabled` is `true` in `infrastructure_config.yaml`. This fixes non-determinism from LLM calls.
    2.  **Simulation Seeding**: Use a fixed simulation seed (`sim_seed.py`) to control randomness in environment dynamics.
    3.  **Golden Master Testing**: Implement and against [`Golden Master`](../scenarios/scenario-validation.md) traces to detect even subtle deviations.
    4.  **External Dependencies**: Minimize external factors. If using live APIs, use `dry_run_mode` or mock services for reproducible runs.

## 5. Real-World Integration Issues

### Issue: Actions not appearing in live marketplace
-   **Problem**: Agent proposes actions, but they don't reflect in the real system.
-   **Solution**:
    1.  **`dry_run_mode`**: First, verify that `safety_constraints.dry_run_mode` is set to `false` in your `integration_config.yaml`.
    2.  **API Credentials**: Double-check that all required API credentials (`AWS_ACCESS_KEY_ID`, `MWS_AUTH_TOKEN`, etc.) are correctly set as environment variables and are valid.
    3.  **Network Connectivity**: Ensure the FBA-Bench environment has network access to the marketplace API endpoints.
    4.  **API Rate Limits**: Check for `RateLimitExceededError` from the marketplace API.
    5.  **Safety Constraint Violations**: Review logs for `SafetyConstraintViolationError` messages. Actions violating constraints (e.g., too large a price change) will be blocked.
    6.  **Marketplace-Specific Errors**: Inspect raw API logs (by enabling `debug_api_logging` in `integration_config.yaml`) for marketplace-specific error codes or messages.

### Issue: Agent makes dangerous/unintended live actions
-   **Problem**: Agent's live actions have negative or unintended consequences.
-   **Solution**:
    1.  **Immediately Enable `dry_run_mode`**: Flip `safety_constraints.dry_run_mode` to `true` in `integration_config.yaml` to halt live interactions.
    2.  **Review Safety Constraints**: Audit and strengthen your `safety_constraints` (e.g., lower `max_daily_price_change_percent`, reduce `manual_approval_threshold_usd`).
    3.  **Manual Approval Workflow**: Ensure your `manual_review_queue_endpoint` is active and monitored for high-impact actions.
    4.  **Automated Rollback**: If configured, investigate why `auto_rollback_on_critical_alert` didn't trigger, or if it did, what the outcome was.
    5.  **Agent Debugging**: Use trace analysis to understand the agent's reasoning leading to the problematic action. Refine prompts or skill logic.

For additional help, refer to the more detailed documentation sections or consider opening an issue on the FBA-Bench GitHub repository.