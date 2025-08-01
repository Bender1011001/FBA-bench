import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List

# Placeholder for FBA-Bench simulation interface
# In a real scenario, this would import the actual simulation components
class FBABenchSimulator:
    def __init__(self):
        self._state = {"time": 0, "inventory": 100, "price": 10.0, "demand": 50, "cash": 1000.0}
        self.episode_steps = 0
        self.max_steps_per_episode = 100

    def reset(self) -> Dict[str, Any]:
        self._state = {"time": 0, "inventory": 100, "price": 10.0, "demand": 50, "cash": 1000.0}
        self.episode_steps = 0
        print("Simulator reset to initial state.")
        return self._state

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.episode_steps += 1
        print(f"Simulator executing action: {action}")
        
        # Simulate market dynamics and agent actions
        current_inventory = self._state["inventory"]
        current_price = self._state["price"]
        current_cash = self._state["cash"]
        current_demand = self._state["demand"]

        # Apply actions
        # For simplicity, let's assume 'set_price' action
        if action.get("type") == "set_price":
            new_price = max(1.0, float(action.get("value", current_price)))
            self._state["price"] = new_price
            print(f"Price set to {new_price}")
        elif action.get("type") == "adjust_inventory":
            inventory_change = int(action.get("value", 0))
            self._state["inventory"] = max(0, current_inventory + inventory_change)
            print(f"Inventory adjusted by {inventory_change}")
        
        # Simulate sales based on demand and price
        units_sold = min(current_demand, self._state["inventory"]) # simplified
        revenue = units_sold * self._state["price"]

        # Update state based on sales and time
        self._state["inventory"] -= units_sold
        self._state["cash"] += revenue
        self._state["time"] += 1
        
        # Simulate demand fluctuations (very basic)
        self._state["demand"] = max(10, current_demand + np.random.randint(-10, 11))

        observation = self._state
        reward = revenue # Simplified reward, based on immediate revenue
        
        done = self.episode_steps >= self.max_steps_per_episode
        truncated = False # Can be used for time limits or other non-terminal conditions
        info = {"units_sold": units_sold, "revenue": revenue}
        
        return observation, reward, done, truncated, info

    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"Current State: {self._state}")
        elif mode == 'ansi':
            return f"Time: {self._state['time']}, Inv: {self._state['inventory']}, Price: {self._state['price']:.2f}, Cash: {self._state['cash']:.2f}"
        else:
            super().render(mode=mode) # Fallback to Gymnasium's default render

    def configure_reward(self, objective: str):
        print(f"Simulator reward configured for objective: {objective}")


class FBABenchRLEnvironment(gym.Env):
    """
    Wraps the FBA-Bench simulation as an OpenAI Gym environment for Reinforcement Learning.
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    def __init__(self, simulator: FBABenchSimulator = None, reward_objective: str = "profit_maximization"):
        super().__init__()
        self.simulator = simulator if simulator else FBABenchSimulator()
        self.reward_objective = reward_objective
        
        # Define observation space (state representation)
        # Example: Box space for numerical simulation states
        self.observation_space = spaces.Dict({
            "time": spaces.Box(0, np.inf, shape=(1,), dtype=np.int32),
            "inventory": spaces.Box(0, np.inf, shape=(1,), dtype=np.int32),
            "price": spaces.Box(0., np.inf, shape=(1,), dtype=np.float32),
            "demand": spaces.Box(0, np.inf, shape=(1,), dtype=np.int32),
            "cash": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        })
        
        # Define action space (mapping RL actions to FBA-Bench commands)
        # Example: Discrete actions for setting price or adjusting inventory
        # Action 0: Keep current price
        # Action 1: Increase price by 10%
        # Action 2: Decrease price by 10%
        # Action 3: Increase inventory by 10 units
        # Action 4: Decrease inventory by 10 units
        self.action_space = spaces.Discrete(5) # Example discrete actions

        self.reward_function_config = {}
        self.configure_reward_function(reward_objective)

        self.current_simulation_state: Dict[str, Any] = {}

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Converts simulation state to an RL-friendly observation format."""
        obs = {k: np.array([v], dtype=self.observation_space[k].dtype)
               for k, v in self.current_simulation_state.items()}
        return obs

    def _get_info(self, simulator_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts additional information from the simulator for logging/debugging."""
        return simulator_info

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Starts a new episode in a deterministic state.
        
        :param seed: Seed for reproducibility.
        :param options: Optional parameters for resetting the environment.
        :return: Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self.current_simulation_state = self.simulator.reset()
        
        observation = self._get_obs()
        info = self._get_info({}) # Initial info
        print("RL Environment reset.")
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes an action and returns the next observation, reward, done flag, and info.
        
        :param action: The action to be taken by the agent (integer mapping to pre-defined actions).
        :return: observation, reward, done (episode finished), truncated (time limit reached), info (for debugging/logging).
        """
        # Map RL action (integer) to FBA-Bench command
        fba_action = self._map_action_to_fba_command(action)
        
        # Execute action in simulator
        obs, reward, terminated, truncated, info = self.simulator.step(fba_action)
        self.current_simulation_state = obs # Update current state after simulator step

        # Calculate reward based on configured function
        calculated_reward = self._calculate_reward(self.current_simulation_state, info)
        
        observation = self._get_obs()
        info = self._get_info(info) # Pass simulator's info through
        
        print(f"RL Step: Action={action}, FBA_Action={fba_action}, Reward={calculated_reward}, Terminated={terminated}, Truncated={truncated}")
        return observation, calculated_reward, terminated, truncated, info

    def render(self, mode: str = 'human'):
        """Visualizes the current simulation state."""
        return self.simulator.render(mode=mode)

    def configure_reward_function(self, objectives: str or List[str]):
        """
        Sets learning objectives and configures the reward function.
        
        :param objectives: A string or list of strings defining the objectives (e.g., "profit_maximization", "inventory_efficiency").
        """
        if isinstance(objectives, str):
            objectives = [objectives]
            
        self.reward_function_config["objectives"] = objectives
        print(f"Reward function configured for objectives: {objectives}")
        self.simulator.configure_reward(objectives[0] if objectives else "default") # Pass to simulator if it also needs config

    def _map_action_to_fba_command(self, rl_action: int) -> Dict[str, Any]:
        """
        Maps a discrete RL action to a specific FBA-Bench command.
        This is a simplified mapping and would be much more complex in a real system.
        """
        current_price = self.current_simulation_state.get("price", 10.0)
        current_inventory = self.current_simulation_state.get("inventory", 100)

        if rl_action == 0: # Keep current price
            return {"type": "set_price", "value": current_price}
        elif rl_action == 1: # Increase price by 10%
            return {"type": "set_price", "value": current_price * 1.1}
        elif rl_action == 2: # Decrease price by 10%
            return {"type": "set_price", "value": current_price * 0.9}
        elif rl_action == 3: # Increase inventory by 10 units
            return {"type": "adjust_inventory", "value": 10}
        elif rl_action == 4: # Decrease inventory by 10 units
            return {"type": "adjust_inventory", "value": -10}
        else:
            raise ValueError(f"Invalid RL action: {rl_action}")

    def _calculate_reward(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """
        Calculates the reward based on the configured objectives.
        This function defines the 'Reward engineering' aspect.
        """
        reward = 0.0
        for objective in self.reward_function_config.get("objectives", []):
            if objective == "profit_maximization":
                # Assume info contains 'revenue' and 'cost' (placeholder for cost)
                revenue = info.get("revenue", 0.0)
                cost = state.get("inventory_cost", 0.0) # Placeholder
                reward += (revenue - cost) # Maximize profit
            elif objective == "revenue_maximization":
                reward += info.get("revenue", 0.0)
            elif objective == "inventory_efficiency":
                # Penalize excess inventory or stockouts
                inventory = state.get("inventory", 0)
                max_inventory = 200 # Example threshold
                if inventory > max_inventory:
                    reward -= (inventory - max_inventory) * 0.1 # Penalty for overstock
                elif inventory <= 10:
                    reward -= 5 # Penalty for potential stockout
            # Add more reward functions as needed for different objectives
            
        return reward

    def get_action_space(self) -> gym.spaces.Space:
        """
        Returns the valid action space for the current environment.
        """
        return self.action_space

    def close(self):
        """Cleans up resources."""
        print("RL Environment closed.")
