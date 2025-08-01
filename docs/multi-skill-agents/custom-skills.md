# Creating Custom Skill Modules

FBA-Bench's multi-skill agent system is designed for extensibility, allowing users to create and integrate their own custom skill modules. This guide will walk you through the process of developing a new skill, from defining its purpose to implementing its logic.

## Skill Module Fundamentals

Every custom skill module must:
1.  **Inherit from `BaseSkill`**: Your skill class must extend [`agents/skill_modules/base_skill.py`](agents/skill_modules/base_skill.py). This provides the necessary interface and logging capabilities.
2.  **Define a `skill_name`**: A unique string identifier for your skill. This is used in the configuration to enable or disable the skill.
3.  **Implement `execute(self, current_state: dict, marketplace_data: dict) -> list[Event]`**: This is the core logic method for your skill.
4.  **Optionally Implement `observe(self, events: list[Event])`**: For reacting to global simulation events.
5.  **Optionally Implement `get_tools(self) -> list[Callable]`**: If your skill uses specific tools callable by an LLM.

## Step-by-Step Guide to Creating a Custom Skill

### Step 1: Create a New Python File
Navigate to the `agents/skill_modules/` directory and create a new Python file for your skill, e.g., [`agents/skill_modules/my_new_skill.py`](agents/skill_modules/my_new_skill.py).

### Step 2: Define Your Skill Class

```python
# agents/skill_modules/my_new_skill.py
from fba_bench.agents.skill_modules.base_skill import BaseSkill
from fba_bench.events import Event # Import relevant event types

class MyNewSkill(BaseSkill):
    def __init__(self, agent_name: str, config: dict):
        super().__init__(agent_name, config)
        self.skill_name = "my_new_skill"
        self.logger.info(f"MyNewSkill initialized for agent: {agent_name}")
        # Initialize any skill-specific state or sub-components here
        self.my_custom_threshold = config.get("custom_threshold", 0.5)

    def execute(self, current_state: dict, marketplace_data: dict) -> list[Event]:
        self.logger.info(f"MyNewSkill for {self.agent_name} executing...")
        proposed_actions = []

        # Example: Simple logic based on current state
        product_price = marketplace_data.get("product_price", {}).get("P001")
        if product_price and product_price > self.my_custom_threshold * 10: # Example logic
            # Propose an action, e.g., a PriceUpdateEvent
            # This requires defining PriceUpdateEvent in events.py or a custom event file
            # For demonstration, let's assume a simple log message as an 'action'
            self.logger.info(f"Product P001 price is high: ${product_price}. Considering adjustment.")
            # In a real scenario, you would create and return an Event object, like:
            # proposed_actions.append(PriceUpdateEvent(product_id="P001", new_price=product_price * 0.9))

        return proposed_actions

    def observe(self, events: list[Event]):
        # This method is called to allow the skill to react to events published on the event bus.
        # Example: Logging a specific event
        for event in events:
            if event.event_type == "DEMAND_SPIKE":
                self.logger.warning(f"MyNewSkill observed a demand spike for {event.product_id}!")

    # Optional: Define tools callable by an LLM if your skill interacts with LLMs
    # def get_tools(self) -> list[Callable]:
    #     def analyze_market_data(product_id: str) -> str:
    #         "Analyzes market data for a given product and returns insights."
    #         # Implement market analysis logic
    #         return f"Market is competitive for {product_id}."
    #     return [analyze_market_data]
```

### Step 3: Configure Your Skill

To make your new skill available to an agent, enable it in the `skill_config.yaml` file (e.g., [`agents/skill_config.py`](agents/skill_config.py)).

```yaml
# Example agents/skill_config.yaml snippet
multi_skill_system:
  enabled_skills:
    - supply_manager
    - marketing_manager
    - my_new_skill # Add your new skill here
  skills: # Optional: define skill-specific configurations
    my_new_skill:
      custom_threshold: 0.6
```

### Step 4: Integrate into an Agent

Your `MultiDomainController` agent will automatically load and activate `MyNewSkill` if it's enabled in the configuration and placed in the `agents/skill_modules/` directory.

```python
from fba_bench.agents.multi_domain_controller import MultiDomainController
from fba_bench.agents.skill_config import SkillConfig
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.tier_0_baseline import tier_0_scenario

# Load skill configuration (ensuring my_new_skill is enabled)
skill_config = SkillConfig.from_yaml("path/to/your/skill_config.yaml") # Load your custom config

agent = MultiDomainController(name="AgentWithCustomSkill", skill_config=skill_config)
scenario_engine = ScenarioEngine(tier_0_scenario)

print("Running simulation with agent utilizing custom skill...")
results = scenario_engine.run_simulation(agent)
print("Simulation complete! Check logs to see MyNewSkill in action.")
```

By following these steps, you can expand FBA-Bench's capabilities with your own specialized agent behaviors. Refer to [`Skill System Overview`](skill-system-overview.md) for a broader architectural understanding.