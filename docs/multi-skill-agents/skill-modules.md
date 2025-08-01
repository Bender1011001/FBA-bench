# Skill Modules

FBA-Bench's Multi-Skill Agent system is composed of individual, specialized `Skill Modules`. Each module encapsulates the logic and tools necessary to manage a specific domain of business operations. This document details the structure of skill modules and provides an overview of the inbuilt modules.

## Skill Module Structure

All skill modules must inherit from the `BaseSkill` class, defined in [`agents/skill_modules/base_skill.py`](agents/skill_modules/base_skill.py). This abstract base class ensures a consistent interface for the `MultiDomainController` to interact with various skills.

Key methods and properties for a skill module:

-   `__init__(self, agent_name: str, config: dict)`: Constructor for the skill, taking the agent's name and its specific configuration.
-   `skill_name`: A static or instance variable string identifying the unique name of the skill (e.g., "supply\_manager").
-   `execute(self, current_state: dict, marketplace_data: dict) -> list[Event]`: The primary method where the skill's core logic resides. It receives the current simulation state and marketplace data, performs its specialized analysis and decision-making, and returns a list of proposed `Event` objects (actions).
-   `observe(self, events: list[Event])`: An optional method for the skill to observe and respond to global simulation events that might be relevant to its domain, even if they didn't directly trigger its `execute` method.
-   `get_tools(self) -> list[Callable]`: An optional method that returns a list of callable tool functions specific to this skill. These tools can then be exposed to an LLM for use within the skill's context.

## Inbuilt Skill Modules

FBA-Bench comes with several pre-defined skill modules, covering common business functions:

### [`Supply Manager Skill (supply_manager.py)`](agents/skill_modules/supply_manager.py)
-   **Responsibility**: Manages inventory levels, raw material procurement, supplier relationships, and logistics.
-   **Key Actions**: Placing orders, adjusting reorder points, managing warehouse capacity.
-   **Example Use Case**: Responding to changes in demand by adjusting inventory orders.

### [`Marketing Manager Skill (marketing_manager.py)`](agents/skill_modules/marketing_manager.py)
-   **Responsibility**: Oversees product promotion, advertising campaigns, pricing strategies, and customer acquisition.
-   **Key Actions**: Adjusting ad spend, launching promotions, changing product prices (in coordination with Financial Analyst).
-   **Example Use Case**: Optimizing ad spend to maximize sales during a peak season.

### [`Customer Service Skill (customer_service.py)`](agents/skill_modules/customer_service.py)
-   **Responsibility**: Handles customer inquiries, resolves complaints, manages reviews, and improves customer satisfaction.
-   **Key Actions**: Responding to negative reviews, issuing refunds, implementing customer loyalty programs.
-   **Example Use Case**: Mitigating reputational damage after a product recall event.

### [`Financial Analyst Skill (financial_analyst.py)`](agents/skill_modules/financial_analyst.py)
-   **Responsibility**: Monitors financial health, manages budgets, forecasts revenue, and makes investment decisions.
-   **Key Actions**: Recommending capital allocation, approving large expenditures, advising on pricing for profitability.
-   **Example Use Case**: Adjusting overall budget allocation based on quarterly financial performance.

## Enabling and Disabling Skill Modules

Skill modules are enabled or disabled via the agent's `skill_config.yaml`. Only enabled skills will be considered by the `MultiDomainController`.

```yaml
# Example skill_config.yaml snippet
multi_skill_system:
  enabled_skills:
    - supply_manager
    - marketing_manager
    - financial_analyst
    #- customer_service # Deactivated for this agent
```

For guidelines on creating your own custom skill modules, refer to [`Custom Skills`](custom-skills.md). For how multiple skills interact and resolve conflicts, see [`Skill Coordination`](skill-coordination.md).