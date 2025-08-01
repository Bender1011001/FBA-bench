# Agent Learning and Adaptation System

This module implements advanced learning capabilities for FBA-Bench agents, enabling them to adapt and improve their performance across simulation runs through reinforcement learning and episodic memory.

## Overview

The learning system consists of three core components:

- **[`EpisodicLearningManager`](episodic_learning.py)**: Persistent storage and retrieval of agent experiences
- **[`FBABenchRLEnvironment`](rl_environment.py)**: OpenAI Gym-compatible reinforcement learning environment
- **[`LearningConfig`](learning_config.py)**: Configuration management for learning parameters

## Key Features

### 🧠 Episodic Learning
- **Experience Storage**: Persistent storage of agent decisions, outcomes, and contexts
- **Pattern Recognition**: Identification of successful strategies and failure modes
- **Knowledge Transfer**: Application of learned experiences to new scenarios
- **Performance Tracking**: Continuous monitoring of agent improvement over time

### 🎯 Reinforcement Learning Integration
- **Gymnasium Compatibility**: Standard RL interface for easy integration with existing frameworks
- **Custom Reward Engineering**: Flexible reward functions tailored to FBA marketplace dynamics
- **State Representation**: Rich observation space including market conditions, competitor behavior, and financial metrics
- **Action Space Mapping**: Translation between high-level business decisions and simulation actions

### ⚙️ Adaptive Configuration
- **Dynamic Learning Rates**: Automatic adjustment based on performance metrics
- **Multi-objective Optimization**: Balance between profit, risk, and market share
- **Safety Constraints**: Preventive measures against dangerous or unrealistic actions
- **Exploration Strategies**: Configurable exploration vs. exploitation balance

## Quick Start

### Basic Learning Setup

```python
from learning.episodic_learning import EpisodicLearningManager
from learning.learning_config import LearningConfig

# Initialize learning components
learning_config = LearningConfig(
    enable_episodic_learning=True,
    reward_function="profit_maximization",
    learning_rate=0.001
)

learning_manager = EpisodicLearningManager()

# Store agent experience
await learning_manager.store_episode_experience(
    agent_id="my_agent",
    episode_data={"actions": [...], "observations": [...]},
    outcomes={"profit": 1500.0, "market_share": 0.15}
)
```

### Reinforcement Learning Environment

```python
from learning.rl_environment import FBABenchRLEnvironment, FBABenchSimulator

# Create RL environment
env = FBABenchRLEnvironment(
    simulator=FBABenchSimulator(),
    reward_objective="profit_maximization"
)

# Standard RL training loop
observation, info = env.reset()
for step in range(1000):
    action = agent.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
```

### CLI Integration

The learning system integrates seamlessly with the experiment CLI:

```bash
# Enable learning during simulations
python experiment_cli.py run sweep.yaml --enable-learning

# Train a specific agent
python experiment_cli.py run --train-agent my_agent_id

# Export trained model
python experiment_cli.py run --export-agent my_agent_id v1.0
```

## Advanced Usage

### Custom Reward Functions

```python
# Define custom reward function
def custom_reward_function(state, action, next_state):
    profit_weight = 0.7
    risk_weight = 0.2
    sustainability_weight = 0.1
    
    profit_score = calculate_profit_score(state, next_state)
    risk_score = calculate_risk_score(action)
    sustainability_score = calculate_sustainability_score(action)
    
    return (profit_weight * profit_score + 
            risk_weight * (1 - risk_score) + 
            sustainability_weight * sustainability_score)

# Configure in learning config
learning_config.reward_function = custom_reward_function
```

### Experience Analysis

```python
# Retrieve and analyze learning progress
progress = await learning_manager.get_learning_progress("my_agent")
print(f"Episodes: {len(progress)}")
print(f"Average reward: {sum(p['reward'] for p in progress) / len(progress)}")

# Export learned strategies
strategy_export = await learning_manager.export_learned_agent("my_agent", "v1.0")
```

## Configuration Options

The [`LearningConfig`](learning_config.py) class provides extensive customization:

```python
learning_config = LearningConfig(
    # Core learning settings
    enable_episodic_learning=True,
    learning_rate=0.001,
    discount_factor=0.95,
    exploration_rate=0.1,
    
    # Buffer and memory settings
    experience_buffer_size=10000,
    batch_size=64,
    memory_consolidation_frequency=100,
    
    # Reward configuration
    reward_function="profit_maximization",
    reward_normalization=True,
    multi_objective_weights={
        "profit": 0.6,
        "risk": 0.2,
        "market_share": 0.2
    },
    
    # Safety and constraints
    safety_constraints_enabled=True,
    max_price_change_per_step=0.1,
    inventory_safety_margin=0.1,
    
    # Training parameters
    training_episodes=1000,
    evaluation_frequency=50,
    save_model_frequency=100
)
```

## Performance Metrics

The learning system tracks various performance metrics:

- **Episode Rewards**: Cumulative reward per episode
- **Learning Progress**: Improvement rate over time
- **Strategy Effectiveness**: Success rate of learned strategies
- **Exploration Efficiency**: Balance between exploration and exploitation
- **Transfer Learning**: Performance on new scenarios using learned knowledge

## Best Practices

### 1. Reward Function Design
- Start with simple reward functions and gradually add complexity
- Ensure rewards are dense enough to provide learning signal
- Avoid reward hacking by testing edge cases
- Balance short-term and long-term objectives

### 2. Experience Management
- Regularly prune old experiences to prevent memory bloat
- Prioritize high-value experiences for faster learning
- Use experience replay for sample efficiency
- Monitor data distribution to prevent overfitting

### 3. Hyperparameter Tuning
- Start with conservative learning rates
- Use learning rate scheduling for better convergence
- Adjust exploration rates based on training progress
- Validate hyperparameters on held-out scenarios

### 4. Safety Considerations
- Always enable safety constraints in live environments
- Test learned policies extensively before deployment
- Monitor agent behavior for unexpected strategies
- Maintain human oversight for critical decisions

## Integration with Other Systems

The learning system integrates with:

- **[Real-World Adapter](../integration/real_world_adapter.py)**: Deploy learned strategies to actual marketplaces
- **[Plugin Framework](../plugins/plugin_framework.py)**: Create learning-enabled community plugins
- **[Scenario Engine](../scenarios/scenario_engine.py)**: Test learned strategies across diverse scenarios
- **[Reproducibility System](../reproducibility/)**: Ensure deterministic learning for research

## Troubleshooting

### Common Issues

**Learning Not Converging**
- Check reward function for appropriate scaling
- Verify exploration rate is not too high/low
- Ensure sufficient training episodes
- Review experience buffer size

**Memory Usage Growing**
- Enable experience buffer pruning
- Reduce buffer size if necessary
- Monitor memory consolidation frequency
- Check for memory leaks in custom components

**Performance Degradation**
- Verify learning rate scheduling
- Check for catastrophic forgetting
- Review safety constraint impacts
- Ensure proper normalization

## Research Applications

The learning system supports various research directions:

- **Multi-Agent Learning**: Competitive and cooperative learning scenarios
- **Transfer Learning**: Knowledge transfer across different market conditions
- **Meta-Learning**: Learning to learn new strategies quickly
- **Continual Learning**: Adaptation to changing market dynamics
- **Interpretable AI**: Understanding learned strategies and decision processes

For more examples and advanced usage patterns, see the [`examples/`](../examples/) directory.