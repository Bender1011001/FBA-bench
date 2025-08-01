# Extending Core FBA-Bench Functionality

FBA-Bench is designed with extensibility in mind, allowing developers to go beyond simple configuration and writing plugins to modify or add core functionalities. This guide explores key areas where you can extend FBA-Bench to suit advanced research needs or unique simulation requirements.

## 1. Adding New Skill Modules and Tools

While `Plugin Development` allows for creating new skills as part of a plugin, you can also directly add new skill modules to the core `agents/skill_modules/` directory if they are fundamental capabilities intended for the main FBA-Bench distribution.

### Steps to Add a New Core Skill Module:
1.  **Create a new file**: Create a Python file in `agents/skill_modules/` (e.g., `agents/skill_modules/my_core_skill.py`).
2.  **Implement `BaseSkill`**: Ensure your new skill class inherits from [`agents/skill_modules/base_skill.py`](agents/skill_modules/base_skill.py) and implements the `execute` method.
3.  **Define `skill_name`**: Set the `skill_name` attribute for your class.
4.  **Register with `SkillFactory` (Implicit)**: The `MultiDomainController` or a core initializer often uses a factory pattern to discover and load skills. Ensure your new skill is discoverable (e.g., by being imported in `agents/skill_modules/__init__.py`).
5.  **Update `SkillConfig`**: Add your new `skill_name` to the default `enabled_skills` in [`agents/skill_config.py`](agents/skill_config.py) or document its manual addition for users.

### Adding New Tools for LLMs
To enable LLMs within agents to use new external capabilities (e.g., a custom API call for weather data, a specific financial calculator), you need to:
1.  **Implement the Tool Function**: Write a Python function that encapsulates the logic.
2.  **Expose to LLM**: Use a mechanism (e.g., function calling/tool definitions supported by the LLM client, or by passing the tool function to the agent's context) to make the tool callable by the LLM. This is often done within a `BaseSkill`'s `get_tools()` method.
    - An example: [`llm_interface/contract.py`](llm_interface/contract.py) defines the schema for LLM-callable tools.

## 2. Modifying Core Cognitive Components

You might want to experiment with alternative implementations of planning, reflection, or memory.

### Replacing `HierarchicalPlanner` or `ReflectionModule`
1.  **Implement a new class**: Create a new class that adheres to the public interface (methods and their signatures) of the existing `HierarchicalPlanner` or `ReflectionModule`.
2.  **Update `AdvancedAgent` (or `CognitiveAgent` factory)**: Modify [`agents/advanced_agent.py`](agents/advanced_agent.py) or a factory method to instantiate your custom class instead of the default, potentially based on a configuration parameter.
3.  **Update `CognitiveConfig`**: Add a configuration option in [`agents/cognitive_config.py`](agents/cognitive_config.py) to enable your custom module.

### Extending Memory Functionality
-   **Custom `MemoryValidator` or `MemoryEnforcer`**: Develop new validation rules or enforcement logic by extending or replacing existing classes in `memory_experiments/` and integrating them into `DualMemoryManager`.
-   **New Storage Backends**: Implement new `long_term_storage_backend` options (e.g., a specific NoSQL database, a custom vector store) by adhering to the storage interface. This would likely involve modifying [`memory_experiments/dual_memory_manager.py`](memory_experiments/dual_memory_manager.py).

## 3. Creating Custom Scenarios beyond YAML

While most scenarios are YAML-defined, very complex or highly dynamic scenarios might require custom Python logic.

-   **Extend `ScenarioEngine`**: Create a custom `ScenarioEngine` sub-class that overrides `load_scenario` or `run_simulation` logic to implement your unique environment dynamics and event injection.
-   **Custom Event Types**: Define new `Event` types in `events.py` or a custom event module to represent novel simulation occurrences.
-   **Custom Marketplace Models**: Develop new or modified marketplace models (`models/`) that simulate more complex economic behaviors, competitor strategies, or consumer reactions. These would be integrated into your custom scenario via the `ScenarioEngine`.
-   **Advanced `DynamicGenerator`**: Extend [`scenarios/dynamic_generator.py`](scenarios/dynamic_generator.py) to create highly specialized scenario generation algorithms suitable for your research.

## 4. Integrating with External Systems

Beyond the standard integrations (like Amazon Seller Central API), you might need to connect FBA-Bench to other external systems.

-   **New Marketplace APIs**: Implement new classes that inherit from `MarketplaceAPI` within `integration/marketplace_apis/` for novel e-commerce platforms. Update `MarketplaceFactory` and `IntegrationConfig` accordingly.
-   **External Data Feeds**: Create modules that fetch real-time data (e.g., stock prices, news feeds) and inject them as events into the simulation.
-   **Custom Observability Sinks**: Extend the `PerformanceMonitor` or `TraceAnalyzer` to export data to proprietary monitoring systems or data warehouses by implementing new data export formats or connectors in `instrumentation/export_utils.py`.

## 5. Modifying Core Event Handling

The `EventBus` (in `event_bus.py` and `infrastructure/distributed_event_bus.py`) is central to FBA-Bench's communication.
-   **Custom Event Processors**: Implement new subscribers to the `EventBus` to introduce novel reactions to specific events without modifying existing agent or simulation logic.
-   **Alternative Event Bus Implementations**: Replace the default Redis/Local Memory event bus with a different messaging system (e.g., RabbitMQ, Google Pub/Sub) by implementing a new `DistributedEventBus` class that adheres to the same interface.

## Best Practices for Extending FBA-Bench

-   **Maintain Modularity**: Keep your extensions as self-contained as possible to minimize impact on the core codebase.
-   **Adhere to Interfaces**: When replacing or extending existing components, ensure your new implementations adhere to the established interfaces (method signatures, expected inputs/outputs) to maintain compatibility.
-   **Comprehensive Testing**: Write unit and integration tests for all your extensions.
-   **Document Thoroughly**: Document your extensions (`docs/development/`) so others can understand, use, and contribute to them.
-   **Engage with Community**: Share your extensions and discuss significant architectural changes with the FBA-Bench community for feedback and potential integration into the main project.