# Multi-Agent Cooperation and Competition Scenarios

FBA-Bench supports complex multi-agent simulations where different agents can operate in cooperative or competitive modes, reflecting real-world business ecosystems. This documentation details how to configure and run scenarios involving multiple intelligent agents.

## Core Concepts

-   **Agent Roles**: Each agent in a multi-agent scenario can be assigned a specific role (e.g., "seller", "supplier", "competitor", "market analyst"), influencing its goals, actions, and interactions.
-   **Cooperative Scenarios**: Agents work together towards a shared objective, requiring communication, joint planning, and resource sharing. Examples: joint ventures, supply chain partnerships.
-   **Competitive Scenarios**: Agents compete against each other for limited resources, market share, or customer attention. Examples: direct competition in a marketplace, market manipulation.
-   **Marketplace Dynamics**: The simulation environment models interactions between multiple agents, including ripple effects of one agent's actions on others.

## How Multi-Agent Scenarios Work

Multi-agent scenarios are defined in scenario YAML files (e.g., [`scenarios/multi_agent/marketplace_ecosystem.yaml`](scenarios/multi_agent/marketplace_ecosystem.yaml)) by specifying multiple agent configurations and their initial states. The `Scenario Engine` (or an orchestrator like the [`Distributed Coordinator`](../infrastructure/distributed-coordinator.md)) manages the concurrent execution and interaction of these agents.

### Key elements in multi-agent scenario YAML:

```yaml
# Example scenarios/multi_agent/my_multi_agent_scenario.yaml
name: "CooperativeVendorPartnership"
description: "Two agents cooperate to maximize overall market coverage."
duration_days: 180

initial_state:
  # ... (initial marketplace conditions) ...

agents:
  - name: "AgentA"
    type: "seller"
    start_capital: 50000
    products: ["product_X"]
    goals:
      - name: "MaximizeProductXSales"
        metric: "sales_product_X"
        target: 1000
        type: "minimum_threshold"
    # Additional agent-specific configs like cognitive_config_path, skill_config_path
    config_profiles:
      cognitive: "default_cognitive_agent"
      skill: "marketing_focused"

  - name: "AgentB"
    type: "distributor"
    start_capital: 30000
    products: ["product_X"]
    goals:
      - name: "EfficientDistribution"
        metric: "inventory_turnover"
        product_id: "product_X"
        target: 12
        type: "minimum_threshold"
    config_profiles:
      cognitive: "planning_heavy_agent"
      skill: "supply_focused"

# Event definitions can now consider interactions between agents
marketplace_events:
  - day: 60
    type: "joint_marketing_opportunity"
    agents_involved: ["AgentA", "AgentB"]
    description: "Combined marketing campaign opportunity. Requires coordinated decision."
```

## Running Multi-Agent Simulations

To run a multi-agent simulation, you typically load the scenario and then set up a mechanism to run multiple agents. In distributed setups, the `Distributed Coordinator` handles this automatically.

```python
# tutorial_multi_agent_simulation.py
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.multi_domain_controller import MultiDomainController # Or AdvancedAgent, depending on scenario
from fba_bench.scenarios.multi_agent.marketplace_ecosystem import marketplace_ecosystem_scenario

# This assumes marketplace_ecosystem_scenario is already loaded from a YAML/Python module
scenario_engine = ScenarioEngine(marketplace_ecosystem_scenario)

# In a multi-agent scenario, the ScenarioEngine itself might instantiate agents
# based on its configuration. Or you might manually instantiate them:
agent_configs_from_scenario = marketplace_ecosystem_scenario.get("agents", [])
agents = []
for agent_cfg in agent_configs_from_scenario:
    # Assuming MultiDomainController is the base for complex agents
    # In a real setup, parse agent_cfg for specific types/configs
    agent = MultiDomainController(name=agent_cfg["name"])
    agents.append(agent)

# For actual execution, use an agent manager or distributed runner that handles multiple agents
print(f"Starting multi-agent simulation for scenario: {scenario_engine.scenario_name}")
# Simplified: in reality, this would involve DistributedCoordinator or AgentManager
# results_per_agent = scenario_engine.run_multi_agent_simulation(agents)
print("Multi-agent simulation setup complete. Execution handled by the Distributed Coordinator.")
print("Refer to docs/infrastructure/distributed-simulation.md for running environments.")
```
## Competition Scenarios and Adversarial Agents

FBA-Bench allows for competitive scenarios where agents vie for market dominance. This can involve predefined competitor agents (often rule-based or using simpler LLMs) or other advanced FBA-Bench agents.

For testing agent robustness against unexpected or malicious behavior, explore the Red Team features documented in [`red_team_framework.md`](../docs/red_team_framework.md). This can simulate adversarial market conditions or competitor attacks.

## Cooperative Scenarios and Communication

In cooperative settings, agents need to share information and coordinate actions. This can be facilitated by:
-   **Shared Memory/State**: Agents accessing a common view of the simulation.
-   **Event-Driven Communication**: Agents publishing events that other agents can subscribe to.
-   **Joint Planning Modules**: Custom modules (outside core FBA-Bench) for multi-agent planning.

For comprehensive details on scenario validation and dynamic generation, refer to [`Scenario Validation`](scenario-validation.md) and [`Dynamic Generation`](dynamic-generation.md).