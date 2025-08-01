# Real-World Integration Example

This directory provides an example of how FBA-Bench agents can be configured for safe interaction with real-world marketplace APIs, or
how to simulate such interactions with built-in safety constraints. This is a critical step for deploying tested agents into live operational environments.

## Features Demonstrated

-   **Safe Deployment**: Agent actions are vetted against configurable safety constraints.
-   **Dry Run Mode**: Test real-world integrations without making actual API calls.
-   **Marketplace API Integration**: Conceptual (and optionally real) interaction with Amazon Seller Central API.
-   **Pre-Flight Validation**: Tools for checking integration setup before deployment.

## Directory Structure

-   `run_real_world_integration_example.py`: The main script to run this integration example.
-   `integration_config.yaml`: Configuration file for the real-world integration system.
-   `live_marketplace_scenario.yaml`: A scenario that might trigger real-world actions.

## How to Run the Example

1.  **Navigate to the Example Directory**:
    ```bash
    cd docs/examples/real-world-integration-example
    ```

2.  **Ensure FBA-Bench Dependencies are Installed**: If you haven't already, install the core FBA-Bench requirements:
    ```bash
    pip install -r ../../../requirements.txt
    ```
    (Adjust path as necessary based on your current working directory relative to the FBA-Bench root)

3.  **Configure Live API Credentials (Optional but Recommended)**:
    For actual (simulated in dry-run mode) API interactions, you'd set your Amazon MWS/Selling Partner API credentials as environment variables.
    
    ```bash
    export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
    export MWS_AUTH_TOKEN="YOUR_MWS_AUTH_TOKEN"
    export SELLER_ID="YOUR_SELLER_ID"
    ```
    _Note: The example `integration_config.yaml` will default to `dry_run_mode: true` to prevent unintended live actions during demonstration._

4.  **Run the Simulation**:
    ```bash
    python run_real_world_integration_example.py
    ```

## Analyzing the Output

-   **Console Output**: The script will indicate whether actions are being sent to a "dry-run" adapter or a live API (if configured and `dry_run_mode` is `false`). It will also report any safety constraint violations.
-   **Trace Files**: A detailed trace of agent actions and the `RealWorldAdapter`'s processing will be saved. Use the `TraceViewer` or `TraceAnalyzer` API ([`docs/observability/trace-analysis.md`](docs/observability/trace-analysis.md)) to see how actions were evaluated against constraints.
-   **Security Logs**: If `debug_api_logging` is enabled in `integration_config.yaml`, you might see logs of the raw API requests and responses (use with caution).

## Critical Safety Note

**NEVER run `dry_run_mode: false` in a production environment without thorough testing, robust monitoring, and comprehensive safety mechanisms in place.** FBA-Bench provides tools to help ensure safety (`safety_constraints` in `integration_config.yaml`, `IntegrationValidator`), but responsibility ultimately lies with the user.

## Customization

-   **`integration_config.yaml`**: This is the most important file for customization. Experiment with:
    -   Toggling `dry_run_mode` (with caution!).
    -   Adjusting `safety_constraints` (e.g., `max_daily_price_change_percent`, `manual_approval_threshold_usd`).
    -   Enabling (if implemented) other `marketplace_adapters`.
    -   Refer to the [`Real-World Integration Configuration`](../../configuration/integration-config.md) guide for all available parameters.
-   **`live_marketplace_scenario.yaml`**: Adjust the scenario elements to provoke specific agent actions that would interact with the real-world system. See the [`Scenario and Curriculum Configuration`](../../configuration/scenario-config.md) guide.
-   **`run_real_world_integration_example.py`**: Modify the script to integrate a different agent or implement more complex pre-deployment validation checks using `IntegrationValidator`.