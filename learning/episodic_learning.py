import os
import json
from collections import defaultdict, deque
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class EpisodeData:
    """Data structure for storing episode information."""
    episode_id: str
    agent_id: str
    states: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    timestamp: str
    metadata: Dict[str, Any]


class ExperienceBuffer:
    """
    Buffer for storing and managing agent experiences.
    
    This class provides a way to store, retrieve, and manage experiences
    for learning purposes.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize the experience buffer with a maximum size."""
        self.max_size = max_size
        self.experiences: List[EpisodeData] = []
    
    def add_experience(self, experience: EpisodeData) -> None:
        """Add an experience to the buffer."""
        self.experiences.append(experience)
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)
    
    def get_experiences(self, agent_id: str = None, limit: int = -1) -> List[EpisodeData]:
        """
        Get experiences from the buffer.
        
        Args:
            agent_id: Optional agent ID to filter experiences
            limit: Maximum number of experiences to return (-1 for all)
            
        Returns:
            List of experiences
        """
        experiences = self.experiences
        if agent_id:
            experiences = [exp for exp in experiences if exp.agent_id == agent_id]
        
        if limit > 0:
            experiences = experiences[-limit:]
            
        return experiences
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.experiences.clear()
    
    def size(self) -> int:
        """Get the current number of experiences in the buffer."""
        return len(self.experiences)


class EpisodicLearningManager:
    """
    Manages the persistent storage, retrieval, and application of agent learning experiences
    across multiple simulation runs (episodes).
    """

    def __init__(self, storage_dir="learning_data"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.agent_experiences: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100)) # Store last 100 episodes
        self.agent_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._load_all_agent_history()

    def _get_agent_file_path(self, agent_id: str) -> str:
        """Helper to get the file path for an agent's experience."""
        return os.path.join(self.storage_dir, f"agent_{agent_id}_experience.json")

    def _load_agent_history(self, agent_id: str):
        """Loads an agent's history from a file."""
        file_path = self._get_agent_file_path(agent_id)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.agent_experiences[agent_id].extend(data.get("experiences", []))
                self.agent_metrics[agent_id] = data.get("metrics", [])
            print(f"Loaded history for agent {agent_id}")

    def _load_all_agent_history(self):
        """Loads history for all agents found in the storage directory."""
        for filename in os.listdir(self.storage_dir):
            if filename.startswith("agent_") and filename.endswith("_experience.json"):
                agent_id = filename.replace("agent_", "").replace("_experience.json", "")
                self._load_agent_history(agent_id)

    def _save_agent_history(self, agent_id: str):
        """Saves an agent's history to a file."""
        file_path = self._get_agent_file_path(agent_id)
        with open(file_path, 'w') as f:
            json.dump({
                "experiences": list(self.agent_experiences[agent_id]),
                "metrics": self.agent_metrics[agent_id]
            }, f, indent=4)
        print(f"Saved history for agent {agent_id}")

    async def store_episode_experience(self, agent_id: str, episode_data: Dict[str, Any], outcomes: Dict[str, Any]):
        """
        Saves learning data for a specific agent after an episode.
        
        :param agent_id: Identifier for the agent.
        :param episode_data: Data from the episode (e.g., states, actions taken, rewards).
        :param outcomes: Key outcomes or summaries of the episode.
        """
        experience = {"episode_data": episode_data, "outcomes": outcomes}
        self.agent_experiences[agent_id].append(experience)
        self._save_agent_history(agent_id)
        print(f"Stored episode experience for agent {agent_id}.")

    async def retrieve_agent_history(self, agent_id: str, num_episodes: int = -1) -> List[Dict[str, Any]]:
        """
        Retrieves past experiences for an agent for learning purposes.
        
        :param agent_id: Identifier for the agent.
        :param num_episodes: Number of most recent episodes to retrieve. Use -1 for all available.
        :return: A list of past episode experiences.
        """
        if agent_id not in self.agent_experiences:
            print(f"No history found for agent {agent_id}.")
            return []
        
        history = list(self.agent_experiences[agent_id])
        if num_episodes == -1 or num_episodes >= len(history):
            return history
        else:
            return history[-num_episodes:] # Return the most recent num_episodes

    async def update_agent_strategy(self, agent_id: str, learnings: Dict[str, Any]):
        """
        Applies improvements to an agent's decision-making based on past outcomes.
        This method would typically involve loading agent's model, applying learnings,
        and saving the updated model. For now, it's a placeholder.
        
        :param agent_id: Identifier for the agent.
        :param learnings: Data representing the improvements (e.g., updated weights, new rules).
        """
        print(f"Applying learnings to agent {agent_id}: {learnings.keys()}")
        # In a real system, this would interact with the agent's internal model/logic.
        # Example: agent_manager.load_agent_model(agent_id)
        #          agent_model.apply_learnings(learnings)
        #          agent_manager.save_agent_model(agent_id, agent_model)
        # Placeholder for demonstration:
        if "strategy_update" in learnings:
            print(f"Agent {agent_id} strategy updated based on: {learnings['strategy_update']}")
        else:
            print(f"Agent {agent_id} received learnings, but no specific strategy update: {learnings}")

    async def track_learning_progress(self, agent_id: str, metrics: Dict[str, Any]):
        """
        Monitors an agent's improvement over multiple episodes.
        
        :param agent_id: Identifier for the agent.
        :param metrics: Dictionary of performance metrics for the current episode/learning step.
        """
        self.agent_metrics[agent_id].append(metrics)
        self._save_agent_history(agent_id)
        print(f"Tracked learning progress for agent {agent_id}: {metrics}")

    async def export_learned_agent(self, agent_id: str, version: str) -> str:
        """
        Saves a trained agent for evaluation or deployment, ensuring clear separation
        from learning mode.
        
        :param agent_id: Identifier for the agent.
        :param version: Version string for the exported agent model.
        :return: Path to the exported agent model.
        """
        export_path = os.path.join(self.storage_dir, f"exported_agent_{agent_id}_v{version}.json")
        # In a real system, this would involve serializing the actual agent model.
        # For demonstration, we'll save a dummy file.
        agent_data = {
            "agent_id": agent_id,
            "version": version,
            "status": "exported",
            "metadata": "This is a placeholder for the actual learned agent model."
        }
        with open(export_path, 'w') as f:
            json.dump(agent_data, f, indent=4)
        print(f"Exported learned agent {agent_id} (version {version}) to {export_path}")
        return export_path