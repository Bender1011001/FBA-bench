# my_simple_agent_plugin/my_simple_skill.py

from fba_bench.agents.skill_modules.base_skill import BaseSkill
from fba_bench.events import Event as FBAEvent # Use alias to avoid naming conflicts if 'Event' is common elsewhere

class MySimpleSkill(BaseSkill):
    """
    A very basic custom skill module for demonstration purposes.
    It simply logs when it executes and observes.
    """
    def __init__(self, agent_name: str, config: dict):
        super().__init__(agent_name, config)
        self.skill_name = "my_simple_skill"
        self.logger.info(f"MySimpleSkill initialized for agent: {agent_name} with config: {config}")
        # You can access skill-specific config here, e.g., self.custom_param = config.get("custom_param", "default")

    def execute(self, current_state: dict, marketplace_data: dict) -> list[FBAEvent]:
        """
        Executes the main logic of the skill.
        Analyzes current_state and marketplace_data, then proposes actions (Events).
        """
        self.logger.info(f"MySimpleSkill for {self.agent_name} executing at day {current_state.get('current_day', 'N/A')}")
        # In a real skill, you would implement complex business logic here,
        # potentially using LLMs or other tools, and generating specific FBAEvents.
        
        # Example: if agent's capital is low, log a warning
        current_capital = current_state.get("financials", {}).get("current_capital", 0.0)
        if current_capital < 1000:
            self.logger.warning(f"MySimpleSkill: Agent {self.agent_name} has low capital: {current_capital}")
            # You could also propose a CapitalInjectionRequestEvent here
            
        return [] # Return a list of proposed actions (Events)

    def observe(self, events: list[FBAEvent]):
        """
        Allows the skill to passively observe global simulation events.
        """
        for event in events:
            if event.event_type == "DEMAND_SPIKE":
                self.logger.info(f"MySimpleSkill observed a DEMAND_SPIKE for product {event.product_id}!")
            elif event.event_type == "ERROR_EVENT":
                self.logger.error(f"MySimpleSkill observed an ERROR_EVENT: {event.message}")

    # Optional: If this skill needs to expose tools for an LLM to call
    # def get_tools(self) -> list[Callable[..., Any]]:
    #     def example_tool_function(param1: str, param2: int) -> str:
    #         "An example tool callable by LLM agents. Does something with param1 and param2."
    #         self.logger.info(f"Example tool called with {param1} and {param2}")
    #         return f"Tool output for {param1} and {param2}"
    #     return [example_tool_function]