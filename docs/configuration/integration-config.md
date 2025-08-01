# Real-World Integration Configuration

This document provides a detailed reference for configuring FBA-Bench's real-world integration system. This includes enabling connections to live marketplaces (like Amazon Seller Central), defining safety constraints, and managing API credentials. Proper configuration is crucial for safe and controlled deployment of agents to production environments.

## Configuration File Location

Real-world integration configurations are typically loaded from YAML files. The primary configuration schema is defined by the `IntegrationConfig` class in [`integration/integration_config.py`](integration/integration_config.py).

## Root-Level Parameters

```yaml
real_world_integration_system:
  enabled: false # Master switch to enable/disable all real-world integration features

  # Nested configurations for sub-systems
  # marketplace_adapters: {...}
  # safety_constraints: {...}
  # deployment_workflow: {...}
```

-   **`enabled`**: (`boolean`, default: `false`)
    -   If `false`, agent actions that would normally interact with live marketplaces will be logged as 'simulated' or 'dry-run' actions and will not reach external APIs. This is a crucial safety switch.

## `marketplace_adapters` Parameters

Defines which real-world marketplaces FBA-Bench can connect to and their specific API settings.

```yaml
marketplace_adapters:
  amazon_seller_central:
    enabled: true
    marketplace_id: "ATVPDKIKX0DER" # Example: US marketplace ID
    api_credentials_env_vars:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - MWS_AUTH_TOKEN
      - SELLER_ID
    api_endpoint: "https://mws.amazonservices.com" # Or Selling Partner API endpoint
    debug_api_logging: false # Log raw API requests/responses
  etsy_shop:
    enabled: false
    api_key_env: "ETSY_API_KEY"
    shop_id: "your_etsy_shop_id"
```

-   Each key under `marketplace_adapters` represents a specific marketplace integration (e.g., `amazon_seller_central`).
-   **`enabled`**: (`boolean`, default: `false`) Per-marketplace toggle.
-   **`api_credentials_env_vars`**: (`list[str]`, required) A list of environment variable names that FBA-Bench should look up to get API credentials for this marketplace. **Highly recommended to use environment variables for sensitive data.**
-   `marketplace_id`, `api_endpoint`, `shop_id`, etc.: Marketplace-specific parameters and endpoints.
-   `debug_api_logging`: (`boolean`, default: `false`) If `true`, enables verbose logging of actual API requests and responses. Use with caution in production.


## `safety_constraints` Parameters

Defines rules and thresholds to prevent unintended or harmful actions when interacting with live marketplaces. See [`Safety Constraints`](../integration/safety-constraints.md) for more details.

```yaml
safety_constraints:
  dry_run_mode: true # All actions are simulated, none are sent to live APIs
  manual_approval_threshold_usd: 5000.00 # Actions with financial impact > this need explicit approval
  max_daily_price_change_percent: 0.15 # Max 15% price change in 24 hours
  max_inventory_restock_units: 5000
  restricted_product_ids: [] # List of product IDs that agents cannot modify pricing or inventory for
  fail_on_violation: true # If true, simulation halts on constraint violation; else, logs warning
```

-   **`dry_run_mode`**: (`boolean`, default: `true`) If `true`, the `RealWorldAdapter` will simulate sending actions but will not make actual API calls. **Highly recommended for testing.**
-   **`manual_approval_threshold_usd`**: (`float`, default: `5000.00`) Any agent action proposing a financial change (e.g., a large order, significant ad spend) exceeding this USD value will require explicit manual approval (if integrated with a manual review workflow).
-   **`max_daily_price_change_percent`**: (`float`, default: `0.10`) Prevents agents from making drastic price changes within a 24-hour period. (e.g., `0.10` for 10% max change).
-   **`max_inventory_restock_units`**: (`integer`, default: `1000`) Limits the maximum quantity of units an agent can order or restock in a single action.
-   **`restricted_product_ids`**: (`list[str]`, default: `[]`) A blacklist of product IDs for which agents are absolutely forbidden from making any operational changes.
-   **`fail_on_violation`**: (`boolean`, default: `true`) If `true`, the simulation will stop immediately if a safety constraint is violated. If `false`, a warning will be logged, and the action will be blocked, but the simulation will continue.

## `deployment_workflow` Parameters

Configures aspects related to deploying agents to live environments, such as monitoring and rollback procedures.

```yaml
deployment_workflow:
  monitoring_integration_enabled: true # Connects to observability system for live agent monitoring
  auto_rollback_on_critical_alert: false # Enable automated rollback if critical alert (e.g., significant profit drop) occurs
  rollback_webhook_url: "https://your-ops.com/rollback" # Webhook to trigger rollback system
  manual_review_queue_endpoint: "https://your-internal-tool.com/review-queue" # Endpoint for manual approval queue
```

-   **`monitoring_integration_enabled`**: (`boolean`, default: `true`) If `true`, ensures that the deployed agent's actions and performance metrics are continuously fed into the FBA-Bench monitoring systems (and potentially external dashboards).
-   **`auto_rollback_on_critical_alert`**: (`boolean`, default: `false`) **Highly sensitive setting.** If `true`, a critical alert (e.g., a rapid, severe drop in profit or surge in errors) will trigger an automated rollback mechanism to revert aggressive agent actions or put the agent into a safe "monitor-only" mode. Requires `rollback_webhook_url`.
-   **`rollback_webhook_url`**: (`str`, optional) A URL endpoint for triggering an external automated rollback system.
-   **`manual_review_queue_endpoint`**: (`str`, optional) An endpoint for a human-in-the-loop system to review and approve/reject agent actions that exceed `manual_approval_threshold_usd`.

## Example Usage

To load and use a custom real-world integration configuration:

```python
from fba_bench.integration.integration_config import IntegrationConfig
from fba_bench.integration.real_world_adapter import RealWorldAdapter

# Load your custom integration configuration file
custom_config_path = "path/to/your/custom_integration_config.yaml"
integration_config = IntegrationConfig.from_yaml(custom_config_path)

# Initialize the RealWorldAdapter with the custom configuration
# This adapter will then process all agent actions destined for external APIs.
real_world_adapter = RealWorldAdapter(config=integration_config)

# You would then pass actions through this adapter, e.g.:
# if real_world_adapter.handle_agent_action(price_update_event):
#     print("Price update sent to live marketplace (or dry-run simulated).")
```

For more details on deploying agents to real marketplaces and managing safety, refer to:
- [`Real-World Integration`](../integration/real-world-integration.md)
- [`Safety Constraints`](../integration/safety-constraints.md)