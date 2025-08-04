"""
Learning module for FBA-Bench.

This module provides learning capabilities for agents, including
episodic learning and reinforcement learning environments.
"""

# Make key classes available at the package level
from .episodic_learning import EpisodicLearningManager
from .rl_environment import FBABenchRLEnvironment, FBABenchSimulator
from .learning_config import LearningConfig

__all__ = ['EpisodicLearningManager', 'FBABenchRLEnvironment', 'FBABenchSimulator', 'LearningConfig']