from dataclasses import dataclass, field
from typing import List
from enum import Enum

class LearningMode(Enum):
    """Learning modes for the agent."""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    IMITATION = "imitation"

@dataclass
class LearningConfig:
    """
    Dataclass for managing learning-related configurations.
    
    Attributes:
        enable_episodic_learning (bool): If True, enables persistent learning between simulation runs.
        learning_mode (str): Specifies the learning paradigm ("reinforcement", "supervised", "imitation").
        experience_buffer_size (int): Max size of the experience replay buffer for episodic learning.
        learning_rate (float): The learning rate for optimization algorithms.
        exploration_strategy (str): The strategy for agent exploration during learning (e.g., "epsilon_greedy", "UCB").
        reward_function (str): Defines the objective for reward calculation (e.g., "profit_maximization", "revenue_growth").
        safety_constraints (List[str]): A list of named safety constraints to apply during learning/deployment.
        real_world_mode (str): The mode for real-world integration ("simulation", "sandbox", "live").
    """
    enable_episodic_learning: bool = False
    learning_mode: str = "reinforcement" # reinforcement, supervised, imitation
    experience_buffer_size: int = 1000
    learning_rate: float = 0.001
    exploration_strategy: str = "epsilon_greedy"
    reward_function: str = "profit_maximization"
    safety_constraints: List[str] = field(default_factory=list)
    real_world_mode: str = "simulation" # simulation, sandbox, live

# Example usage:
if __name__ == "__main__":
    default_config = LearningConfig()
    print(f"Default Learning Config: {default_config}")

    rl_config = LearningConfig(
        enable_episodic_learning=True,
        learning_mode="reinforcement",
        experience_buffer_size=5000,
        learning_rate=0.0005,
        exploration_strategy="boltzmann",
        reward_function="customer_satisfaction",
        safety_constraints=["max_price_drop", "min_inventory_level"],
        real_world_mode="sandbox"
    )
    print(f"\nRL Sandbox Config: {rl_config}")

    live_config = LearningConfig(
        enable_episodic_learning=True,
        learning_mode="reinforcement",
        reward_function="profit_maximization",
        safety_constraints=["strict_price_bounds", "no_negative_inventory_changes"],
        real_world_mode="live"
    )
    print(f"\nLive Deployment Config: {live_config}")