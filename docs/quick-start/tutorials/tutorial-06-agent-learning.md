# Tutorial 6: Training Agents with Episodic Learning

This tutorial explores FBA-Bench's agent learning capabilities, focusing on episodic learning and integration with reinforcement learning environments for continuous improvement.

## Episodic Learning

FBA-Bench agents can learn and adapt across multiple simulation episodes. This means an agent's insights, strategies, or even skill parameters can persist and evolve between runs.

```python
# tutorial_episodic_learning.py
from fba_bench.learning.episodic_learning import EpisodicLearningManager
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.tier_0_baseline import tier_0_scenario
from fba_bench.agents.advanced_agent import AdvancedAgent # Or your learning-capable agent

# Initialize the episodic learning manager
# This manager is responsible for saving/loading agent states, memories, and learned parameters
learning_manager = EpisodicLearningManager(
    agent_id="MyLearningAgent",
    storage_path="./learning_data", # Directory to save learning artifacts
    learning_rate=0.01 # Example learning rate for parameter adjustments
)

# Load or create an agent that supports episodic learning
agent = AdvancedAgent(name="MyLearningAgent") # Ensure AdvancedAgent has learning integration

# Simulate multiple episodes
num_episodes = 3
for episode in range(num_episodes):
    print(f"\n--- Running Episode {episode + 1} ---")
    
    # Load previous learning state if available (for subsequent episodes)
    if episode > 0:
        learning_manager.load_agent_state(agent)

    scenario_engine = ScenarioEngine(tier_0_scenario)
    results = scenario_engine.run_simulation(agent)
    
    # After each episode, process results and update agent's learnable parameters
    learning_manager.process_episode_results(agent, results, episode_number=episode)
    
    # Save the current learning state for the next episode
    learning_manager.save_agent_state(agent)

    print(f"Episode {episode + 1} complete. Agent performance for this episode: {results['metrics']['net_profit'] if 'net_profit' in results.get('metrics', {}) else 'N/A'}")

print("\nEpisodic learning simulation finished. Agent has potentially learned across episodes.")
```

## Reinforcement Learning (RL) Environment Integration

FBA-Bench can be adapted to serve as an OpenAI Gym-compatible environment, allowing you to train agents using standard RL algorithms and frameworks.

```python
# tutorial_rl_environment.py
from fba_bench.learning.rl_environment import FBABenchGymEnv
# from stable_baselines3 import PPO # Example RL library
from fba_bench.scenarios.tier_0_baseline import tier_0_scenario

# Create an FBA-Bench RL environment
env = FBABenchGymEnv(scenario_config=tier_0_scenario)

# Example: (pseudo-code) Train an RL agent
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save("fba_bench_rl_agent")

print("FBA-Bench can be integrated with RL frameworks like Stable Baselines3.")
print("Refer to docs/learning-system/reinforcement-learning.md for detailed setup.")
```

## Safety Considerations: Learning vs. Evaluation Mode

It's crucial to distinguish between learning and evaluation modes. FBA-Bench allows you to configure agents to use learned parameters only during specific phases or to disable learning features entirely for controlled evaluation.

Refer to [`docs/learning-system/safety-considerations.md`](docs/learning-system/safety-considerations.md) for guidelines on managing learning in different simulation contexts.