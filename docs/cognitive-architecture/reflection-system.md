# Reflection System

FBA-Bench's structured reflection system enables agents to introspect, learn from past experiences, and generate actionable insights. This module is crucial for continuous improvement and adaptive behavior in complex simulation environments.

## Core Concepts

-   **Reflection Points**: Predefined moments in the simulation (e.g., end of day, end of quarter, after a critical event) where the agent pauses to reflect.
-   **Contextual Review**: The agent reviews relevant historical data, actions taken, and outcomes achieved since the last reflection.
-   **Discrepancy Identification**: Agents are prompted to identify mismatches between expected and actual outcomes, or between their plans and executed actions.
-   **Insight Generation**: Based on identified discrepancies and reviewed context, the agent generates insights, lessons learned, or potential improvements.
-   **Knowledge Integration**: Generated insights can be integrated back into the agent's memory for future decisions or used to adjust planning parameters.

## How it Works

The Reflection Module (implemented in [`memory_experiments/reflection_module.py`](memory_experiments/reflection_module.py)) operates by:

1.  **Triggering Reflection**: The main simulation loop or agent controller triggers reflection at configured intervals or events.
2.  **Context Assembly**: It collects relevant internal agent state, simulation events, and marketplace data from the designated reflection period.
3.  **LLM-Powered Analysis**: The assembled context is provided to an LLM (specified in the agent configuration) which is prompted to perform structured analysis, identify anomalies, and synthesize insights based on predefined reflection prompts.
4. **Insight Capture**: The LLM's generated insights are parsed, validated, and stored in the agent's memory system.
5. **Feedback Loop**: Insights can optionally be used to update the agent's internal models, adjust planning parameters, or modify future behavior.

## Configuration

Reflection system behavior is configured within the agent's cognitive settings, typically via `cognitive_config.yaml`.

```yaml
# Example cognitive_config.yaml snippet
reflection:
  enabled: true
  frequency: "end_of_quarter" # "end_of_day", "end_of_quarter", "after_critical_event"
  insight_generation: true # Enable LLM to generate insights
  feedback_loop_enabled: true # Whether insights feed back into planning/decision-making
  max_reflection_tokens: 2000 # Limit for LLM reflection prompt/response
```

## Example Usage

While reflection is seamlessly integrated into the [`AdvancedAgent`](agents/advanced_agent.py), you can observe its effects via the simulation traces and logs or manually invoke it for testing.

```python
from fba_bench.memory_experiments.reflection_module import ReflectionModule
from fba_bench.agents.cognitive_config import CognitiveConfig

# Create a sample config
reflection_config = CognitiveConfig(
    reflection={
        "enabled": True,
        "frequency": "end_of_quarter",
        "insight_generation": True
    }
).reflection_config # Access the specific reflection config

# In a real simulation, `agent_memory` and `simulation_context` would be populated
# with actual data by the simulation engine or agent controller.
agent_memory = ["Previous plan: Increase sales by 10%. Actual sales: 5% increase."]
simulation_context = {"quarter_results": {"sales_increase": 0.05, "profit_margin": 0.15}}

# Initialize Reflection Module (requires an LLM client, typically handled by the agent)
# For this example, we'll indicate it's a placeholder.
class MockLLMClient:
    def chat_completion(self, messages, max_tokens, temperature):
        return {"choices": [{"message": {"content": "Insight: Sales growth was lower than expected due to new competitor. Need to adjust marketing."}}]}

mock_llm_client = MockLLMClient()

# The ReflectionModule expects a way to get relevant data
class MockAgentState:
    def get_memory_context(self): return agent_memory
    def get_simulation_context(self): return simulation_context
    def get_llm_client(self): return mock_llm_client
    def store_insight(self, insight): print(f"Stored insight: {insight}")

mock_agent_state = MockAgentState()

reflection_module = ReflectionModule(agent_name="MyReflectiveAgent", config=reflection_config, agent_state=mock_agent_state)

print("Performing reflection...")
insights = reflection_module.perform_reflection()
print("Generated Insights:", insights)

# In a live agent, these insights would be automatically integrated.
```

For more details on memory validation and consistency, refer to the [`Memory Integration`](memory-integration.md) documentation.