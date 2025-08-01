# Tutorial 7: Deploying Agents to Real Marketplaces

This tutorial outlines the process and considerations for safely integrating FBA-Bench agents with real-world marketplace APIs, emphasizing safety constraints and best practices.

## Overview of Real-World Integration

FBA-Bench provides a `real_world_adapter` module that acts as a secure intermediary between your simulated agents and live marketplace APIs (e.g., Amazon Seller Central). This ensures that agent actions are vetted and conform to predefined safety constraints before execution.

## Setting Up Amazon Seller Central Integration

1.  **Obtain API Credentials**: Register for a developer account with Amazon Seller Central and obtain your Seller ID, MWS Auth Token, and AWS Access Key/Secret Key. Store these securely, preferably as environment variables.

    ```bash
    export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
    export MWS_AUTH_TOKEN="YOUR_MWS_AUTH_TOKEN"
    export SELLER_ID="YOUR_SELLER_ID"
    ```

2.  **Configure FBA-Bench**: Update your `integration_config.yaml` (or similar configuration file) to enable Amazon Seller Central integration and specify safety parameters.

    ```yaml
    # integration/integration_config.yaml
    real_world_integration:
      enabled: true
      marketplace_adapter: "amazon_seller_central"
      safety_constraints:
        max_daily_price_change_percent: 0.10 # Max 10% price change per day
        max_inventory_restock_units: 1000
        manual_approval_threshold_usd: 5000 # Actions above this value require manual review
      api_credentials_env_vars:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - MWS_AUTH_TOKEN
        - SELLER_ID
    ```

3.  **Run with Integration Enabled**:
    The system will automatically instantiate the `RealWorldAdapter` and `AmazonSellerCentralAPI` based on your configuration. Your agent will interact with the simulation as usual, but its "real-world" actions will be routed through the adapter.

    ```python
    # tutorial_real_world_integration.py
    from fba_bench.integration.real_world_adapter import RealWorldAdapter
    from fba_bench.scenarios.scenario_engine import ScenarioEngine
    from fba_bench.scenarios.tier_0_baseline import tier_0_scenario # Or a more complex scenario
    from fba_bench.agents.advanced_agent import AdvancedAgent
    from fba_bench.integration.integration_validator import IntegrationValidator # For pre-flight checks

    # This example assumes you have an agent capable of making real-world actions
    # such as PriceUpdateAction, InventoryRestockAction, etc.
    agent = AdvancedAgent(name="LiveMarketplaceAgent")
    scenario_engine = ScenarioEngine(tier_0_scenario)

    # The RealWorldAdapter is usually instantiated by the main simulation runner
    # based on configuration. For demonstration, you could manually integrate:
    # adapter = RealWorldAdapter(config.real_world_integration_config)
    # scenario_engine.register_action_handler(adapter.handle_agent_action)

    # Perform pre-flight validation
    validator = IntegrationValidator()
    if not validator.validate_setup():
        print("Real-world integration setup validation failed. Please check logs.")
        # Exit or provide user feedback
    else:
        print("Real-world integration setup validated. Running simulation with live integration...")
        print("NOTE: No actual API calls will be made in this example. Real calls depend on your agent's actions.")
        # In a real setup, agent actions (e.g., price changes) would be sent to the adapter
        results = scenario_engine.run_simulation(agent)
        print("Simulation with real-world integration complete.")
    ```

## Safety Constraints and Risk Management

FBA-Bench's integration layer includes critical safety mechanisms to prevent unintended or harmful actions in live environments:
-   **Configurable Thresholds**: Define limits for price changes, inventory adjustments, ad spend, etc.
-   **Manual Approval**: Actions exceeding a financial threshold can trigger a manual review process (e.g., via a connected dashboard or alert system).
-   **Dry Run Mode**: Execute actions without actually committing to the marketplace APIs (useful for testing).

For complete details on configuring safety constraints, supported marketplace APIs, deployment workflows, and troubleshooting, refer to the [`docs/integration/`](docs/integration/) documentation.