import unittest
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from learning.episodic_learning import EpisodicLearning
from learning.reinforcement_learning import ReinforcementLearning
from learning.curriculum_learning import CurriculumLearning
from learning.meta_learning import MetaLearning


class TestEpisodicLearning(unittest.TestCase):
    """Test suite for the EpisodicLearning class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.episodic_learning = EpisodicLearning()
    
    def test_episodic_learning_initialization(self):
        """Test that the episodic learning initializes correctly."""
        self.assertIsNotNone(self.episodic_learning)
        self.assertEqual(len(self.episodic_learning._episodes), 0)
        self.assertEqual(len(self.episodic_learning._memories), 0)
    
    def test_record_episode(self):
        """Test recording an episode."""
        episode_data = {
            "episode_id": "episode1",
            "agent_id": "agent1",
            "scenario_id": "scenario1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(seconds=60),
            "actions": [
                {"type": "decision", "data": {"price": 10.0}, "timestamp": datetime.now()},
                {"type": "action", "data": {"quantity": 100}, "timestamp": datetime.now()}
            ],
            "outcomes": {"profit": 1000.0, "market_share": 0.25},
            "metrics": {"accuracy": 0.85, "efficiency": 0.90}
        }
        
        episode_id = self.episodic_learning.record_episode(episode_data)
        
        self.assertEqual(episode_id, "episode1")
        self.assertIn("episode1", self.episodic_learning._episodes)
        self.assertEqual(self.episodic_learning._episodes["episode1"]["agent_id"], "agent1")
        self.assertEqual(self.episodic_learning._episodes["episode1"]["outcomes"]["profit"], 1000.0)
    
    def test_retrieve_episode(self):
        """Test retrieving an episode."""
        episode_data = {
            "episode_id": "episode1",
            "agent_id": "agent1",
            "scenario_id": "scenario1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(seconds=60),
            "actions": [
                {"type": "decision", "data": {"price": 10.0}, "timestamp": datetime.now()}
            ],
            "outcomes": {"profit": 1000.0},
            "metrics": {"accuracy": 0.85}
        }
        
        self.episodic_learning.record_episode(episode_data)
        
        episode = self.episodic_learning.retrieve_episode("episode1")
        
        self.assertIsNotNone(episode)
        self.assertEqual(episode["episode_id"], "episode1")
        self.assertEqual(episode["agent_id"], "agent1")
        self.assertEqual(episode["outcomes"]["profit"], 1000.0)
    
    def test_find_similar_episodes(self):
        """Test finding similar episodes."""
        # Record multiple episodes
        episode1_data = {
            "episode_id": "episode1",
            "agent_id": "agent1",
            "scenario_id": "scenario1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(seconds=60),
            "actions": [
                {"type": "decision", "data": {"price": 10.0}, "timestamp": datetime.now()}
            ],
            "outcomes": {"profit": 1000.0},
            "metrics": {"accuracy": 0.85}
        }
        
        episode2_data = {
            "episode_id": "episode2",
            "agent_id": "agent1",
            "scenario_id": "scenario1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(seconds=60),
            "actions": [
                {"type": "decision", "data": {"price": 12.0}, "timestamp": datetime.now()}
            ],
            "outcomes": {"profit": 1200.0},
            "metrics": {"accuracy": 0.90}
        }
        
        self.episodic_learning.record_episode(episode1_data)
        self.episodic_learning.record_episode(episode2_data)
        
        # Find similar episodes
        query_episode = {
            "actions": [
                {"type": "decision", "data": {"price": 11.0}}
            ]
        }
        
        similar_episodes = self.episodic_learning.find_similar_episodes(query_episode, threshold=0.8)
        
        self.assertEqual(len(similar_episodes), 2)
        self.assertTrue(any(e["episode_id"] == "episode1" for e in similar_episodes))
        self.assertTrue(any(e["episode_id"] == "episode2" for e in similar_episodes))
    
    def test_extract_lessons(self):
        """Test extracting lessons from episodes."""
        # Record an episode
        episode_data = {
            "episode_id": "episode1",
            "agent_id": "agent1",
            "scenario_id": "scenario1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(seconds=60),
            "actions": [
                {"type": "decision", "data": {"price": 10.0}, "timestamp": datetime.now()},
                {"type": "action", "data": {"quantity": 100}, "timestamp": datetime.now()}
            ],
            "outcomes": {"profit": 1000.0, "market_share": 0.25},
            "metrics": {"accuracy": 0.85, "efficiency": 0.90}
        }
        
        self.episodic_learning.record_episode(episode_data)
        
        # Extract lessons
        lessons = self.episodic_learning.extract_lessons("episode1")
        
        self.assertIsNotNone(lessons)
        self.assertIn("patterns", lessons)
        self.assertIn("insights", lessons)
        self.assertIn("recommendations", lessons)
    
    def test_get_episode_statistics(self):
        """Test getting episode statistics."""
        # Record multiple episodes
        episode1_data = {
            "episode_id": "episode1",
            "agent_id": "agent1",
            "scenario_id": "scenario1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(seconds=60),
            "actions": [
                {"type": "decision", "data": {"price": 10.0}, "timestamp": datetime.now()}
            ],
            "outcomes": {"profit": 1000.0},
            "metrics": {"accuracy": 0.85}
        }
        
        episode2_data = {
            "episode_id": "episode2",
            "agent_id": "agent1",
            "scenario_id": "scenario1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(seconds=60),
            "actions": [
                {"type": "decision", "data": {"price": 12.0}, "timestamp": datetime.now()}
            ],
            "outcomes": {"profit": 1200.0},
            "metrics": {"accuracy": 0.90}
        }
        
        self.episodic_learning.record_episode(episode1_data)
        self.episodic_learning.record_episode(episode2_data)
        
        # Get statistics
        stats = self.episodic_learning.get_episode_statistics("agent1")
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["avg_profit"], 1100.0)
        self.assertEqual(stats["avg_accuracy"], 0.875)


class TestReinforcementLearning(unittest.TestCase):
    """Test suite for the ReinforcementLearning class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reinforcement_learning = ReinforcementLearning(
            state_space_size=10,
            action_space_size=5,
            learning_rate=0.01,
            discount_factor=0.99,
            exploration_rate=0.1
        )
    
    def test_reinforcement_learning_initialization(self):
        """Test that the reinforcement learning initializes correctly."""
        self.assertIsNotNone(self.reinforcement_learning)
        self.assertEqual(self.reinforcement_learning._state_space_size, 10)
        self.assertEqual(self.reinforcement_learning._action_space_size, 5)
        self.assertEqual(self.reinforcement_learning._learning_rate, 0.01)
        self.assertEqual(self.reinforcement_learning._discount_factor, 0.99)
        self.assertEqual(self.reinforcement_learning._exploration_rate, 0.1)
        self.assertEqual(self.reinforcement_learning._q_table.shape, (10, 5))
    
    def test_select_action(self):
        """Test selecting an action."""
        state = 5
        
        # Mock random choice for exploration
        with patch('numpy.random.random') as mock_random, \
             patch('numpy.random.choice') as mock_choice:
            
            mock_random.return_value = 0.05  # Less than exploration rate
            mock_choice.return_value = 2
            
            action = self.reinforcement_learning.select_action(state)
            
            self.assertEqual(action, 2)
            mock_random.assert_called_once()
            mock_choice.assert_called_once()
    
    def test_update_q_table(self):
        """Test updating the Q-table."""
        state = 5
        action = 2
        reward = 10.0
        next_state = 7
        
        # Get initial Q-value
        initial_q_value = self.reinforcement_learning._q_table[state, action]
        
        # Update Q-table
        self.reinforcement_learning.update_q_table(state, action, reward, next_state)
        
        # Get updated Q-value
        updated_q_value = self.reinforcement_learning._q_table[state, action]
        
        # Verify the Q-value was updated
        self.assertGreater(updated_q_value, initial_q_value)
    
    def test_decay_exploration_rate(self):
        """Test decaying the exploration rate."""
        initial_exploration_rate = self.reinforcement_learning._exploration_rate
        
        # Decay exploration rate
        self.reinforcement_learning.decay_exploration_rate(decay_rate=0.99)
        
        # Verify the exploration rate was decayed
        self.assertLess(self.reinforcement_learning._exploration_rate, initial_exploration_rate)
    
    def test_get_policy(self):
        """Test getting the policy."""
        policy = self.reinforcement_learning.get_policy()
        
        self.assertIsNotNone(policy)
        self.assertEqual(len(policy), self.reinforcement_learning._state_space_size)
        self.assertTrue(all(0 <= action < self.reinforcement_learning._action_space_size for action in policy))
    
    def test_save_load_model(self):
        """Test saving and loading the model."""
        # Update Q-table
        self.reinforcement_learning.update_q_table(5, 2, 10.0, 7)
        
        # Save model
        with patch('numpy.save') as mock_save:
            self.reinforcement_learning.save_model("test_model.npy")
            mock_save.assert_called_once()
        
        # Load model
        with patch('numpy.load') as mock_load:
            mock_load.return_value = self.reinforcement_learning._q_table
            
            new_rl = ReinforcementLearning(
                state_space_size=10,
                action_space_size=5
            )
            new_rl.load_model("test_model.npy")
            
            mock_load.assert_called_once()
            np.testing.assert_array_equal(new_rl._q_table, self.reinforcement_learning._q_table)


class TestCurriculumLearning(unittest.TestCase):
    """Test suite for the CurriculumLearning class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.curriculum_learning = CurriculumLearning()
    
    def test_curriculum_learning_initialization(self):
        """Test that the curriculum learning initializes correctly."""
        self.assertIsNotNone(self.curriculum_learning)
        self.assertEqual(len(self.curriculum_learning._curriculum), 0)
        self.assertEqual(len(self.curriculum_learning._student_progress), 0)
    
    def test_add_curriculum_level(self):
        """Test adding a curriculum level."""
        level_data = {
            "level_id": "level1",
            "name": "Basic Pricing",
            "description": "Learn basic pricing strategies",
            "scenarios": ["scenario1", "scenario2"],
            "prerequisites": [],
            "success_criteria": {"accuracy": 0.8, "efficiency": 0.7},
            "difficulty": 1
        }
        
        level_id = self.curriculum_learning.add_curriculum_level(level_data)
        
        self.assertEqual(level_id, "level1")
        self.assertIn("level1", self.curriculum_learning._curriculum)
        self.assertEqual(self.curriculum_learning._curriculum["level1"]["name"], "Basic Pricing")
        self.assertEqual(self.curriculum_learning._curriculum["level1"]["difficulty"], 1)
    
    def test_get_curriculum_level(self):
        """Test getting a curriculum level."""
        level_data = {
            "level_id": "level1",
            "name": "Basic Pricing",
            "description": "Learn basic pricing strategies",
            "scenarios": ["scenario1", "scenario2"],
            "prerequisites": [],
            "success_criteria": {"accuracy": 0.8, "efficiency": 0.7},
            "difficulty": 1
        }
        
        self.curriculum_learning.add_curriculum_level(level_data)
        
        level = self.curriculum_learning.get_curriculum_level("level1")
        
        self.assertIsNotNone(level)
        self.assertEqual(level["level_id"], "level1")
        self.assertEqual(level["name"], "Basic Pricing")
        self.assertEqual(level["difficulty"], 1)
    
    def test_update_student_progress(self):
        """Test updating student progress."""
        # Add a curriculum level
        level_data = {
            "level_id": "level1",
            "name": "Basic Pricing",
            "description": "Learn basic pricing strategies",
            "scenarios": ["scenario1", "scenario2"],
            "prerequisites": [],
            "success_criteria": {"accuracy": 0.8, "efficiency": 0.7},
            "difficulty": 1
        }
        
        self.curriculum_learning.add_curriculum_level(level_data)
        
        # Update student progress
        progress_data = {
            "student_id": "student1",
            "level_id": "level1",
            "scenario_results": {
                "scenario1": {"accuracy": 0.85, "efficiency": 0.75},
                "scenario2": {"accuracy": 0.90, "efficiency": 0.80}
            },
            "completion_time": datetime.now(),
            "overall_performance": {"accuracy": 0.875, "efficiency": 0.775}
        }
        
        progress_id = self.curriculum_learning.update_student_progress(progress_data)
        
        self.assertIsNotNone(progress_id)
        self.assertIn("student1", self.curriculum_learning._student_progress)
        self.assertIn("level1", self.curriculum_learning._student_progress["student1"])
        self.assertEqual(
            self.curriculum_learning._student_progress["student1"]["level1"]["overall_performance"]["accuracy"],
            0.875
        )
    
    def test_get_next_level(self):
        """Test getting the next level for a student."""
        # Add multiple curriculum levels
        level1_data = {
            "level_id": "level1",
            "name": "Basic Pricing",
            "description": "Learn basic pricing strategies",
            "scenarios": ["scenario1", "scenario2"],
            "prerequisites": [],
            "success_criteria": {"accuracy": 0.8, "efficiency": 0.7},
            "difficulty": 1
        }
        
        level2_data = {
            "level_id": "level2",
            "name": "Advanced Pricing",
            "description": "Learn advanced pricing strategies",
            "scenarios": ["scenario3", "scenario4"],
            "prerequisites": ["level1"],
            "success_criteria": {"accuracy": 0.9, "efficiency": 0.8},
            "difficulty": 2
        }
        
        self.curriculum_learning.add_curriculum_level(level1_data)
        self.curriculum_learning.add_curriculum_level(level2_data)
        
        # Update student progress for level1
        progress_data = {
            "student_id": "student1",
            "level_id": "level1",
            "scenario_results": {
                "scenario1": {"accuracy": 0.85, "efficiency": 0.75},
                "scenario2": {"accuracy": 0.90, "efficiency": 0.80}
            },
            "completion_time": datetime.now(),
            "overall_performance": {"accuracy": 0.875, "efficiency": 0.775}
        }
        
        self.curriculum_learning.update_student_progress(progress_data)
        
        # Get next level
        next_level = self.curriculum_learning.get_next_level("student1")
        
        self.assertIsNotNone(next_level)
        self.assertEqual(next_level["level_id"], "level2")
        self.assertEqual(next_level["name"], "Advanced Pricing")
    
    def test_get_student_progress(self):
        """Test getting student progress."""
        # Add a curriculum level
        level_data = {
            "level_id": "level1",
            "name": "Basic Pricing",
            "description": "Learn basic pricing strategies",
            "scenarios": ["scenario1", "scenario2"],
            "prerequisites": [],
            "success_criteria": {"accuracy": 0.8, "efficiency": 0.7},
            "difficulty": 1
        }
        
        self.curriculum_learning.add_curriculum_level(level_data)
        
        # Update student progress
        progress_data = {
            "student_id": "student1",
            "level_id": "level1",
            "scenario_results": {
                "scenario1": {"accuracy": 0.85, "efficiency": 0.75},
                "scenario2": {"accuracy": 0.90, "efficiency": 0.80}
            },
            "completion_time": datetime.now(),
            "overall_performance": {"accuracy": 0.875, "efficiency": 0.775}
        }
        
        self.curriculum_learning.update_student_progress(progress_data)
        
        # Get student progress
        progress = self.curriculum_learning.get_student_progress("student1")
        
        self.assertIsNotNone(progress)
        self.assertIn("level1", progress)
        self.assertEqual(progress["level1"]["overall_performance"]["accuracy"], 0.875)


class TestMetaLearning(unittest.TestCase):
    """Test suite for the MetaLearning class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.meta_learning = MetaLearning()
    
    def test_meta_learning_initialization(self):
        """Test that the meta learning initializes correctly."""
        self.assertIsNotNone(self.meta_learning)
        self.assertEqual(len(self.meta_learning._meta_knowledge), 0)
        self.assertEqual(len(self.meta_learning._learning_strategies), 0)
    
    def test_record_learning_experience(self):
        """Test recording a learning experience."""
        experience_data = {
            "experience_id": "exp1",
            "task_type": "pricing",
            "strategy_used": "reinforcement_learning",
            "hyperparameters": {"learning_rate": 0.01, "discount_factor": 0.99},
            "performance": {"accuracy": 0.85, "efficiency": 0.90},
            "learning_time": 3600,  # seconds
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        experience_id = self.meta_learning.record_learning_experience(experience_data)
        
        self.assertEqual(experience_id, "exp1")
        self.assertIn("exp1", self.meta_learning._meta_knowledge)
        self.assertEqual(self.meta_learning._meta_knowledge["exp1"]["task_type"], "pricing")
        self.assertEqual(self.meta_learning._meta_knowledge["exp1"]["strategy_used"], "reinforcement_learning")
    
    def test_recommend_learning_strategy(self):
        """Test recommending a learning strategy."""
        # Record multiple learning experiences
        exp1_data = {
            "experience_id": "exp1",
            "task_type": "pricing",
            "strategy_used": "reinforcement_learning",
            "hyperparameters": {"learning_rate": 0.01, "discount_factor": 0.99},
            "performance": {"accuracy": 0.85, "efficiency": 0.90},
            "learning_time": 3600,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        exp2_data = {
            "experience_id": "exp2",
            "task_type": "pricing",
            "strategy_used": "supervised_learning",
            "hyperparameters": {"learning_rate": 0.001, "batch_size": 32},
            "performance": {"accuracy": 0.80, "efficiency": 0.85},
            "learning_time": 1800,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        self.meta_learning.record_learning_experience(exp1_data)
        self.meta_learning.record_learning_experience(exp2_data)
        
        # Recommend learning strategy
        task_description = {
            "task_type": "pricing",
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        recommendation = self.meta_learning.recommend_learning_strategy(task_description)
        
        self.assertIsNotNone(recommendation)
        self.assertIn("strategy", recommendation)
        self.assertIn("hyperparameters", recommendation)
        self.assertIn("expected_performance", recommendation)
        # Should recommend reinforcement_learning as it performed better
        self.assertEqual(recommendation["strategy"], "reinforcement_learning")
    
    def test_adapt_hyperparameters(self):
        """Test adapting hyperparameters based on meta-knowledge."""
        # Record learning experiences
        exp1_data = {
            "experience_id": "exp1",
            "task_type": "pricing",
            "strategy_used": "reinforcement_learning",
            "hyperparameters": {"learning_rate": 0.01, "discount_factor": 0.99},
            "performance": {"accuracy": 0.85, "efficiency": 0.90},
            "learning_time": 3600,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        exp2_data = {
            "experience_id": "exp2",
            "task_type": "pricing",
            "strategy_used": "reinforcement_learning",
            "hyperparameters": {"learning_rate": 0.02, "discount_factor": 0.99},
            "performance": {"accuracy": 0.90, "efficiency": 0.95},
            "learning_time": 3600,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        self.meta_learning.record_learning_experience(exp1_data)
        self.meta_learning.record_learning_experience(exp2_data)
        
        # Adapt hyperparameters
        current_hyperparameters = {"learning_rate": 0.01, "discount_factor": 0.99}
        task_description = {
            "task_type": "pricing",
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        adapted_hyperparameters = self.meta_learning.adapt_hyperparameters(
            current_hyperparameters,
            task_description
        )
        
        self.assertIsNotNone(adapted_hyperparameters)
        self.assertIn("learning_rate", adapted_hyperparameters)
        self.assertIn("discount_factor", adapted_hyperparameters)
        # Should recommend higher learning rate as it performed better
        self.assertGreater(adapted_hyperparameters["learning_rate"], current_hyperparameters["learning_rate"])
    
    def test_extract_meta_patterns(self):
        """Test extracting meta-patterns from learning experiences."""
        # Record learning experiences
        exp1_data = {
            "experience_id": "exp1",
            "task_type": "pricing",
            "strategy_used": "reinforcement_learning",
            "hyperparameters": {"learning_rate": 0.01, "discount_factor": 0.99},
            "performance": {"accuracy": 0.85, "efficiency": 0.90},
            "learning_time": 3600,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        exp2_data = {
            "experience_id": "exp2",
            "task_type": "inventory",
            "strategy_used": "reinforcement_learning",
            "hyperparameters": {"learning_rate": 0.01, "discount_factor": 0.99},
            "performance": {"accuracy": 0.80, "efficiency": 0.85},
            "learning_time": 3600,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        self.meta_learning.record_learning_experience(exp1_data)
        self.meta_learning.record_learning_experience(exp2_data)
        
        # Extract meta-patterns
        patterns = self.meta_learning.extract_meta_patterns()
        
        self.assertIsNotNone(patterns)
        self.assertIn("strategy_performance", patterns)
        self.assertIn("hyperparameter_importance", patterns)
        self.assertIn("task_characteristics", patterns)
    
    def test_get_meta_knowledge_summary(self):
        """Test getting a summary of meta-knowledge."""
        # Record learning experiences
        exp1_data = {
            "experience_id": "exp1",
            "task_type": "pricing",
            "strategy_used": "reinforcement_learning",
            "hyperparameters": {"learning_rate": 0.01, "discount_factor": 0.99},
            "performance": {"accuracy": 0.85, "efficiency": 0.90},
            "learning_time": 3600,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        exp2_data = {
            "experience_id": "exp2",
            "task_type": "inventory",
            "strategy_used": "supervised_learning",
            "hyperparameters": {"learning_rate": 0.001, "batch_size": 32},
            "performance": {"accuracy": 0.80, "efficiency": 0.85},
            "learning_time": 1800,
            "metadata": {"dataset_size": 1000, "complexity": "medium"}
        }
        
        self.meta_learning.record_learning_experience(exp1_data)
        self.meta_learning.record_learning_experience(exp2_data)
        
        # Get meta-knowledge summary
        summary = self.meta_learning.get_meta_knowledge_summary()
        
        self.assertIsNotNone(summary)
        self.assertIn("total_experiences", summary)
        self.assertIn("task_types", summary)
        self.assertIn("strategies_used", summary)
        self.assertIn("performance_summary", summary)
        self.assertEqual(summary["total_experiences"], 2)
        self.assertIn("pricing", summary["task_types"])
        self.assertIn("inventory", summary["task_types"])
        self.assertIn("reinforcement_learning", summary["strategies_used"])
        self.assertIn("supervised_learning", summary["strategies_used"])


if __name__ == '__main__':
    unittest.main()