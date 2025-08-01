"""
Learning System Example for FBA-Bench

This example demonstrates how to use FBA-Bench's agent learning and adaptation
capabilities, including episodic learning, reinforcement learning integration,
and persistent experience storage.

Key Features Demonstrated:
- Episodic Learning Manager usage
- RL Environment integration
- Learning configuration management
- Experience storage and retrieval
- Agent performance tracking
- Model export and import
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# FBA-Bench core imports
from learning.episodic_learning import EpisodicLearningManager, EpisodeData, ExperienceBuffer
from learning.rl_environment import FBABenchRLEnvironment, RLConfig
from learning.learning_config import LearningConfig, LearningMode
from experiment_cli import ExperimentRunner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningExample:
    """Comprehensive example of FBA-Bench learning capabilities."""
    
    def __init__(self):
        """Initialize the learning example."""
        self.learning_manager = None
        self.rl_environment = None
        self.config = None
        self.experiment_runner = None
        
    async def setup_learning_system(self) -> None:
        """Set up the learning system components."""
        logger.info("Setting up learning system...")
        
        # Create learning configuration
        self.config = LearningConfig(
            mode=LearningMode.REINFORCEMENT_LEARNING,
            enable_episodic_memory=True,
            rl_config={
                'algorithm': 'PPO',
                'learning_rate': 0.001,
                'batch_size': 64,
                'epsilon': 0.1,
                'gamma': 0.99,
                'episodes': 1000
            },
            memory_config={
                'max_episodes': 10000,
                'max_buffer_size': 100000,
                'similarity_threshold': 0.8,
                'compression_enabled': True
            },
            export_config={
                'export_path': './models/trained_agent',
                'export_format': 'pytorch',
                'include_metadata': True
            }
        )
        
        # Initialize episodic learning manager
        self.learning_manager = EpisodicLearningManager(
            config=self.config.memory_config
        )
        
        # Initialize RL environment
        rl_config = RLConfig(
            algorithm=self.config.rl_config['algorithm'],
            learning_rate=self.config.rl_config['learning_rate'],
            batch_size=self.config.rl_config['batch_size'],
            epsilon=self.config.rl_config['epsilon'],
            gamma=self.config.rl_config['gamma']
        )
        
        self.rl_environment = FBABenchRLEnvironment(config=rl_config)
        
        logger.info("Learning system setup complete")

    async def demonstrate_episodic_learning(self) -> None:
        """Demonstrate episodic learning capabilities."""
        logger.info("\n=== Episodic Learning Demonstration ===")
        
        # Create sample episode data
        sample_episodes = [
            EpisodeData(
                episode_id="ep_001",
                scenario_config={"market_type": "competitive", "difficulty": "moderate"},
                actions=[
                    {"action": "price_change", "product_id": "PROD001", "new_price": 29.99},
                    {"action": "inventory_order", "product_id": "PROD001", "quantity": 100}
                ],
                observations=[
                    {"market_price": 25.00, "competitor_count": 5, "demand": 150},
                    {"market_price": 27.50, "competitor_count": 5, "demand": 120}
                ],
                rewards=[10.5, 15.2],
                outcome_metrics={
                    "total_revenue": 2999.0,
                    "profit_margin": 0.15,
                    "customer_satisfaction": 0.85
                },
                performance_score=8.7,
                metadata={
                    "agent_strategy": "aggressive_pricing",
                    "market_conditions": "stable"
                }
            ),
            EpisodeData(
                episode_id="ep_002",
                scenario_config={"market_type": "volatile", "difficulty": "hard"},
                actions=[
                    {"action": "price_change", "product_id": "PROD002", "new_price": 45.99},
                    {"action": "marketing_campaign", "budget": 500, "duration": 7}
                ],
                observations=[
                    {"market_price": 40.00, "competitor_count": 8, "demand": 200},
                    {"market_price": 42.50, "competitor_count": 7, "demand": 180}
                ],
                rewards=[5.3, 12.8],
                outcome_metrics={
                    "total_revenue": 4599.0,
                    "profit_margin": 0.12,
                    "customer_satisfaction": 0.78
                },
                performance_score=7.2,
                metadata={
                    "agent_strategy": "defensive_pricing",
                    "market_conditions": "volatile"
                }
            )
        ]
        
        # Store episodes
        for episode in sample_episodes:
            await self.learning_manager.store_episode(episode)
            logger.info(f"Stored episode: {episode.episode_id}")
        
        # Retrieve similar episodes
        query_scenario = {"market_type": "competitive", "difficulty": "moderate"}
        similar_episodes = await self.learning_manager.retrieve_similar_episodes(
            scenario_config=query_scenario,
            limit=5
        )
        
        logger.info(f"Found {len(similar_episodes)} similar episodes for query scenario")
        for episode in similar_episodes:
            logger.info(f"  - {episode.episode_id}: score={episode.performance_score}")
        
        # Get performance insights
        insights = await self.learning_manager.get_performance_insights()
        logger.info("Performance insights:")
        logger.info(f"  - Average performance: {insights['average_performance']:.2f}")
        logger.info(f"  - Best performance: {insights['best_performance']:.2f}")
        logger.info(f"  - Total episodes: {insights['total_episodes']}")
        logger.info(f"  - Success rate: {insights['success_rate']:.2%}")

    async def demonstrate_rl_training(self) -> None:
        """Demonstrate reinforcement learning training."""
        logger.info("\n=== Reinforcement Learning Demonstration ===")
        
        # Create a simple market scenario for training
        scenario_config = {
            "market_type": "standard",
            "products": ["PROD001", "PROD002"],
            "competitors": 3,
            "simulation_days": 30
        }
        
        # Reset environment
        initial_state = await self.rl_environment.reset(scenario_config)
        logger.info(f"Environment reset. Initial state shape: {len(initial_state)}")
        
        # Training loop simulation
        total_reward = 0
        episode_rewards = []
        
        for episode in range(10):  # Reduced for demo purposes
            state = await self.rl_environment.reset(scenario_config)
            episode_reward = 0
            
            for step in range(20):  # 20 steps per episode
                # Sample action (in real training, this would come from RL agent)
                action = self._sample_action()
                
                # Take step in environment
                next_state, reward, done, info = await self.rl_environment.step(action)
                
                episode_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
            
            logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        average_reward = total_reward / len(episode_rewards)
        logger.info(f"Training complete. Average reward: {average_reward:.2f}")
        
        # Get training metrics
        metrics = await self.rl_environment.get_training_metrics()
        logger.info("Training metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  - {key}: {value:.4f}")
            else:
                logger.info(f"  - {key}: {value}")

    async def demonstrate_experience_buffer(self) -> None:
        """Demonstrate experience buffer functionality."""
        logger.info("\n=== Experience Buffer Demonstration ===")
        
        # Create experience buffer
        buffer = ExperienceBuffer(max_size=1000)
        
        # Add sample experiences
        experiences = []
        for i in range(50):
            experience = {
                'state': [i * 0.1, (i + 1) * 0.1, (i + 2) * 0.1],
                'action': i % 4,  # 4 possible actions
                'reward': (i % 10) / 10.0,
                'next_state': [(i + 1) * 0.1, (i + 2) * 0.1, (i + 3) * 0.1],
                'done': (i % 20) == 19,
                'timestamp': i
            }
            experiences.append(experience)
            buffer.add(experience)
        
        logger.info(f"Added {len(experiences)} experiences to buffer")
        logger.info(f"Buffer size: {len(buffer)}")
        logger.info(f"Buffer capacity: {buffer.max_size}")
        
        # Sample from buffer
        batch_size = 10
        sample = buffer.sample(batch_size)
        logger.info(f"Sampled batch of {len(sample)} experiences")
        
        # Clear buffer
        buffer.clear()
        logger.info(f"Buffer cleared. New size: {len(buffer)}")

    async def demonstrate_learning_integration(self) -> None:
        """Demonstrate integration with experiment runner."""
        logger.info("\n=== Learning Integration Demonstration ===")
        
        # Create experiment configuration with learning enabled
        experiment_config = {
            "scenario": {
                "name": "learning_demo",
                "config_path": "scenarios/tier_1_moderate.yaml"
            },
            "agents": [
                {
                    "name": "learning_agent",
                    "class": "BasicAgent",
                    "config": {
                        "strategy": "adaptive",
                        "learning_enabled": True
                    }
                }
            ],
            "learning": {
                "enabled": True,
                "mode": "reinforcement_learning",
                "config": self.config.to_dict()
            },
            "metrics": [
                "revenue",
                "profit",
                "market_share",
                "learning_progress"
            ]
        }
        
        # Initialize experiment runner with learning
        self.experiment_runner = ExperimentRunner(
            config=experiment_config,
            learning_manager=self.learning_manager
        )
        
        logger.info("Experiment runner initialized with learning capabilities")
        
        # Simulate running a learning-enabled experiment
        logger.info("Running learning-enabled experiment simulation...")
        
        # This would normally run a full experiment
        # For demo purposes, we'll simulate the learning aspects
        training_progress = {
            "episode": 0,
            "total_episodes": 100,
            "current_reward": 0.0,
            "average_reward": 0.0,
            "epsilon": 1.0,
            "learning_rate": 0.001
        }
        
        for episode in range(5):  # Simulate 5 episodes
            training_progress["episode"] = episode + 1
            training_progress["current_reward"] = 10.0 + episode * 2.5
            training_progress["average_reward"] = (training_progress["average_reward"] * episode + 
                                                  training_progress["current_reward"]) / (episode + 1)
            training_progress["epsilon"] = max(0.1, 1.0 - episode * 0.2)
            
            logger.info(f"Episode {training_progress['episode']}: "
                       f"Reward={training_progress['current_reward']:.2f}, "
                       f"Avg={training_progress['average_reward']:.2f}, "
                       f"ε={training_progress['epsilon']:.2f}")

    async def demonstrate_model_export(self) -> None:
        """Demonstrate model export and import functionality."""
        logger.info("\n=== Model Export/Import Demonstration ===")
        
        # Create sample model data
        model_data = {
            "model_type": "PPO",
            "architecture": {
                "input_size": 10,
                "hidden_layers": [64, 32],
                "output_size": 4
            },
            "training_config": self.config.rl_config,
            "performance_metrics": {
                "average_reward": 15.7,
                "best_reward": 23.4,
                "training_episodes": 1000,
                "convergence_episode": 750
            },
            "weights": "model_weights_placeholder",  # In real scenario, this would be actual weights
            "metadata": {
                "trained_on": "2024-01-15",
                "scenario_types": ["competitive", "volatile"],
                "version": "1.0.0"
            }
        }
        
        # Export model
        export_path = Path(self.config.export_config['export_path'])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path.with_suffix('.json'), 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model exported to: {export_path.with_suffix('.json')}")
        
        # Import model (simulate)
        with open(export_path.with_suffix('.json'), 'r') as f:
            imported_model = json.load(f)
        
        logger.info("Model imported successfully")
        logger.info(f"Model type: {imported_model['model_type']}")
        logger.info(f"Performance: {imported_model['performance_metrics']['average_reward']:.2f}")

    def _sample_action(self) -> Dict[str, Any]:
        """Sample a random action for demonstration purposes."""
        import random
        
        action_types = ["price_change", "inventory_order", "marketing_campaign"]
        action_type = random.choice(action_types)
        
        if action_type == "price_change":
            return {
                "type": "price_change",
                "product_id": random.choice(["PROD001", "PROD002"]),
                "price_change": random.uniform(-5.0, 5.0)
            }
        elif action_type == "inventory_order":
            return {
                "type": "inventory_order",
                "product_id": random.choice(["PROD001", "PROD002"]),
                "quantity": random.randint(10, 100)
            }
        else:  # marketing_campaign
            return {
                "type": "marketing_campaign",
                "budget": random.randint(100, 1000),
                "duration": random.randint(3, 14)
            }

    async def run_complete_example(self) -> None:
        """Run the complete learning example."""
        logger.info("=== FBA-Bench Learning System Example ===\n")
        
        try:
            # Setup
            await self.setup_learning_system()
            
            # Run demonstrations
            await self.demonstrate_episodic_learning()
            await self.demonstrate_rl_training()
            await self.demonstrate_experience_buffer()
            await self.demonstrate_learning_integration()
            await self.demonstrate_model_export()
            
            logger.info("\n=== Example Complete ===")
            logger.info("All learning system features demonstrated successfully!")
            
        except Exception as e:
            logger.error(f"Error during example execution: {e}")
            raise


async def main():
    """Main function to run the learning example."""
    example = LearningExample()
    await example.run_complete_example()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())