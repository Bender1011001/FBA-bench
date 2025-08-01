# Master Configuration Guide

FBA-Bench's behavior is extensively customizable through a set of configuration files. This master guide provides an overview of the global configuration philosophy, common file formats, and best practices for managing settings across the entire FBA-Bench ecosystem.

## 1. Configuration Philosophy

FBA-Bench uses a layered, modular configuration approach:
-   **Defaults**: Each major system (Cognitive, Multi-Skill, Infrastructure, Scenarios, etc.) has sensible default configurations hardcoded or provided in default YAML files within its module (e.g., [`agents/cognitive_config.py`](agents/cognitive_config.py)).
-   **YAML-Based Overrides**: Users can override default settings by providing custom YAML files. This is the primary method for customizing FBA-Bench for specific experiments or agent behaviors.
-   **Environment Variables**: Key settings, especially sensitive ones like API keys, can be overridden by environment variables, offering flexibility for deployment and CI/CD pipelines.
-   **Programmatic Overrides**: For advanced users, configurations can be modified directly in Python scripts before initializing components.

## 2. Configuration File Formats and Locations

Most configurations are defined in YAML (`.yaml` or `.yml`) files due to their human-readability and direct mapping to structured data.

### Primary Configuration Directories:
-   **`agents/configs/`**: Contains configurations for cognitive architecture (e.g., how hierarchical planning, reflection, and memory are set up). Example: `default_cognitive_config.yaml`.
-   **`agents/skill_modules/` and `agents/`**: Specifically, [`agents/skill_config.py`](agents/skill_config.py) defines the schema for multi-skill agent settings, including which skills are enabled and their coordination strategies.
-   **`infrastructure/`**: Includes configurations for scalability, LLM batching, and distributed simulation (`infrastructure/scalability_config.py`).
-   **`scenarios/`**: Defines the environment, events, and goals for simulations (e.g., `tier_0_baseline.yaml`, `business_types/`).
-   **`observability/`**: Settings for monitoring, logging, and alerting (`observability/observability_config.py`).
-   **`integration/`**: Configures real-world API integrations and safety constraints (`integration/integration_config.py`).
-   **`learning/`**: Controls agent learning and adaptation parameters (`learning/learning_config.py`).
-   **`constraints/`**: Defines tier-specific constraints for benchmarking (`constraints/constraint_config.py`, `constraints/tier_configs/`).
-   **`llm_interface/`**: Contains prompt templates and settings for LLM interaction (`llm_interface/prompt_templates.py`).

## 3. Environment Variable Reference

For sensitive information or deployment-specific overrides, environment variables are recommended. FBA-Bench components are designed to check for relevant environment variables before falling back to file-based or default configurations.

Common environment variables:
-   `OPENROUTER_API_KEY`: Your API key for OpenRouter LLM service.
-   `ANTHROPIC_API_KEY`: Your API key for Anthropic LLM service.
-   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MWS_AUTH_TOKEN`, `SELLER_ID`: For Amazon Seller Central integration.
-   `REDIS_HOST`, `REDIS_PORT`: For configuring the distributed event bus.
-   `FBA_BENCH_MODE`: To set the global operational mode (e.g., "production", "development").

## 4. Configuration Validation and Troubleshooting

FBA-Bench applies schema validation (using libraries like Pydantic) to its configuration files. If your YAML file has incorrect keys, missing required values, or invalid data types, the system will raise `ValidationError` errors during startup.

### Common Troubleshooting Steps:
1.  **Check YAML Syntax**: Ensure your YAML is correctly formatted (indentation is crucial!).
2.  **Refer to Schema**: Compare your custom configuration against the relevant `config.py` file (e.g., `CognitiveConfig` in `agents/cognitive_config.py`) to ensure you're using the correct keys and types.
3.  **Error Messages**: Pay close attention to `ValidationError` messages; they usually point to the exact issue.
4.  **Environment Variables**: Double-check that environment variables are correctly set and accessible to the FBA-Bench process.

## 5. System-Specific Configuration Guides

For detailed options and examples for each major system, refer to:
-   [`Cognitive System Configuration`](cognitive-config.md)
-   [`Multi-Skill Agent Configuration`](skill-config.md)
-   [`Infrastructure Configuration`](infrastructure-config.md)
-   [`Scenario and Curriculum Configuration`](scenario-config.md)
-   [`Monitoring and Observability Configuration`](observability-config.md)
-   [`Real-World Integration Configuration`](integration-config.md)
-   [`Agent Learning System Configuration`](learning-config.md)