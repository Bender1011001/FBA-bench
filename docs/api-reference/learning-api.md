# Agent Learning System API Reference

This document provides a comprehensive API reference for FBA-Bench's agent learning system, including `EpisodicLearningManager` and `FBABenchGymEnv` for reinforcement learning integration.

## 1. `EpisodicLearningManager`

Manages persistent learning across simulation episodes, allowing agents to retain and adapt strategies, memories, and parameters.

-   **Module**: [`learning/episodic_learning.py`](learning/episodic_learning.py)
-   **Class**: `EpisodicLearningManager`

### Constructor

`__init__(self, agent_id: str, storage_path: str = "./learning_data", learning_rate: float = 0.001, config: dict = None)`

-   **`agent_id`**: (`str`, required) A unique identifier for the learning agent.
-   **`storage_path`**: (`str`, optional) Directory to save and load learning artifacts (agent state, learned parameters, insights). Default is `./learning_data`.
-   **`learning_rate`**: (`float`, optional) A factor influencing how strongly an agent's parameters are updated based on new experience. Default is `0.001`.
-   **`config`**: (`dict`, optional) Additional configuration for the learning manager.

### Key Methods

#### `load_agent_state(self, agent: Any) -> bool`
Loads the previously saved learning state (parameters, memories, etc.) into the provided agent instance.

-   **`agent`**: (`Any`, required) The agent instance to load state into (must implement necessary `load_state` method).
-   **Returns**: `bool` - `True` if state was loaded, `False` otherwise (e.g., no saved state found).

#### `save_agent_state(self, agent: Any)`
Saves the current learning state of the agent to persistent storage.

-   **`agent`**: (`Any`, required) The agent instance whose state is to be saved (must implement necessary `save_state` method).

#### `process_episode_results(self, agent: Any, results: dict, episode_number: int)`
Processes the outcomes of a completed simulation episode, extracting insights, evaluating performance, and triggering learning updates.

-   **`agent`**: (`Any`, required) The agent instance that completed the episode.
-   **`results`**: (`dict`, required) The full results dictionary from the completed simulation episode.
-   **`episode_number`**: (`int`, required) The sequential number of the current episode.

#### `get_learning_history(self) -> list[dict]`
Retrieves a history of performance and insights recorded across all episodes for this agent.

-   **Returns**: `list[dict]` - A list of dictionaries, each representing an episode's learning summary.

## 2. `FBABenchGymEnv`

Provides an OpenAI Gym-compatible environment interface for FBA-Bench simulations, enabling integration with standard reinforcement learning frameworks.

-   **Module**: [`learning/rl_environment.py`](learning/rl_environment.py)
-   **Class**: `FBABenchGymEnv` (inherits from `gym.Env` or similar)

### Constructor

`__init__(self, scenario_config: dict, agent_type: Any = AdvancedAgent, reward_function: Callable = None, cost_function: Callable = None, observation_space_params: dict = None, action_space_params: dict = None)`

-   **`scenario_config`**: (`dict`, required) The configuration dictionary for the FBA-Bench scenario to be used as the environment.
-   **`agent_type`**: (`Any`, optional) The type of FBA-Bench agent class to instantiate and control within the environment. Defaults to `AdvancedAgent`.
-   **`reward_function`**: (`Callable`, optional) A callable function `(current_state: dict, actions: list[Event], metrics_delta: dict) -> float` that calculates the reward for the RL agent. If `None`, uses a default simple reward (e.g., based on net profit).
-   **`cost_function`**: (`Callable`, optional) A callable `(llm_token_cost: float, compute_cost: float) -> float` to penalize LLM/compute costs in the reward signal.
-   **`observation_space_params`**: (`dict`, optional) Parameters to configure the Gym observation space.
-   **`action_space_params`**: (`dict`, optional) Parameters to configure the Gym action space.

### Key Methods (Standard Gym API)

#### `step(self, action: Any) -> Tuple[Any, float, bool, dict]`
Executes one time step in the simulation based on the agent's `action`.

-   **`action`**: (`Any`, required) The action taken by the RL agent, typically a structured representation that maps to FBA-Bench `Event` objects.
-   **Returns**: `Tuple[Any, float, bool, dict]` - `observation`, `reward`, `done`, `info`.
    -   `observation`: The new state of the environment.
    -   `reward`: The reward obtained from the last action.
    -   `done`: `True` if the episode has ended.
    -   `info`: A dictionary for debugging or additional information.

#### `reset(self, seed: int = None, options: dict = None) -> Tuple[Any, dict]`
Resets the environment to its initial state for a new episode.

-   **`seed`**: (`int`, optional) Seed for reproducibility.
-   **`options`**: (`dict`, optional) Additional options for resetting.
-   **Returns**: `Tuple[Any, dict]` - `initial_observation`, `info`.

#### `render(self, mode: str = 'human')`
(Optional) Renders the environment state. For FBA-Bench, this might involve updating the frontend dashboard.