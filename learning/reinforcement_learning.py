"""
Reinforcement Learning (tabular Q-learning) implementation used by tests.

This class provides a production-ready, deterministic tabular Q-learning agent with:
- Epsilon-greedy action selection (exploration_rate)
- Online Q-table updates with learning_rate and discount_factor
- Deterministic save/load via NumPy arrays
- Policy extraction (greedy argmax per state)
- Exploration rate decay utility

It is intentionally framework-agnostic and suitable for unit testing and
simple environments with discrete state/action spaces.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class ReinforcementLearning:
    """
    Tabular Q-learning agent for discrete state/action spaces.

    Parameters
    ----------
    state_space_size : int
        Number of discrete states.
    action_space_size : int
        Number of discrete actions.
    learning_rate : float, optional
        Alpha parameter for Q-learning updates (default 0.1).
    discount_factor : float, optional
        Gamma parameter for future reward discounting (default 0.99).
    exploration_rate : float, optional
        Epsilon parameter for epsilon-greedy action selection (default 0.1).
    random_seed : Optional[int], optional
        Seed for numpy RNG to ensure determinism in tests, by default None.
    """

    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.1,
        random_seed: Optional[int] = None,
    ) -> None:
        if state_space_size <= 0:
            raise ValueError("state_space_size must be > 0")
        if action_space_size <= 0:
            raise ValueError("action_space_size must be > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0.0 <= discount_factor <= 1.0):
            raise ValueError("discount_factor must be in [0, 1]")
        if not (0.0 <= exploration_rate <= 1.0):
            raise ValueError("exploration_rate must be in [0, 1]")

        # RNG for deterministic behavior when desired
        self._rng = np.random.default_rng(random_seed)

        self._state_space_size = int(state_space_size)
        self._action_space_size = int(action_space_size)
        self._learning_rate = float(learning_rate)
        self._discount_factor = float(discount_factor)
        self._exploration_rate = float(exploration_rate)

        # Initialize Q-table to zeros
        self._q_table: np.ndarray = np.zeros(
            (self._state_space_size, self._action_space_size), dtype=np.float64
        )

    def select_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.

        With probability `exploration_rate`, choose a random action.
        Otherwise choose the greedy action argmax(Q[state]).

        Parameters
        ----------
        state : int
            Discrete state index in [0, state_space_size).

        Returns
        -------
        int
            Selected action index in [0, action_space_size).
        """
        self._validate_state(state)
        # Explore
        if self._rng.random() < self._exploration_rate:
            return int(self._rng.integers(0, self._action_space_size))
        # Exploit (greedy)
        return int(np.argmax(self._q_table[state]))

    def update_q_table(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Q-learning update rule:
          Q[s,a] = Q[s,a] + alpha * (reward + gamma * max_a' Q[s', a'] - Q[s,a])

        Parameters
        ----------
        state : int
            Current state index.
        action : int
            Action taken at state.
        reward : float
            Observed reward.
        next_state : int
            Next state index after taking action.
        """
        self._validate_state(state)
        self._validate_state(next_state)
        self._validate_action(action)

        current_q = self._q_table[state, action]
        best_next_q = float(np.max(self._q_table[next_state]))
        td_target = reward + self._discount_factor * best_next_q
        td_error = td_target - current_q

        self._q_table[state, action] = current_q + self._learning_rate * td_error

    def decay_exploration_rate(self, decay_rate: float = 0.99, min_epsilon: float = 0.01) -> None:
        """
        Exponentially decay epsilon for epsilon-greedy policy.

        Parameters
        ----------
        decay_rate : float, optional
            Multiplicative decay each call, by default 0.99.
        min_epsilon : float, optional
            Lower bound for epsilon, by default 0.01.
        """
        if not (0.0 < decay_rate <= 1.0):
            raise ValueError("decay_rate must be in (0, 1]")
        if not (0.0 <= min_epsilon <= 1.0):
            raise ValueError("min_epsilon must be in [0, 1]")

        self._exploration_rate = max(min_epsilon, self._exploration_rate * decay_rate)

    def get_policy(self) -> List[int]:
        """
        Extract a greedy policy from the Q-table.

        Returns
        -------
        List[int]
            For each state, the action index that maximizes Q[state].
        """
        # Argmax along actions for each state
        greedy_actions = np.argmax(self._q_table, axis=1)
        return [int(a) for a in greedy_actions]

    def save_model(self, file_path: str) -> None:
        """
        Persist the Q-table to disk using NumPy savez (single array for simplicity).

        Parameters
        ----------
        file_path : str
            Target path for the saved NumPy array (e.g., "model.npy").
        """
        np.save(file_path, self._q_table)

    def load_model(self, file_path: str) -> None:
        """
        Load Q-table from disk. Validates the shape matches current state/action sizes.

        Parameters
        ----------
        file_path : str
            Source path for the NumPy array saved via save_model().
        """
        q = np.load(file_path, allow_pickle=False)
        if not isinstance(q, np.ndarray):
            raise ValueError("Loaded model is not a NumPy ndarray")
        if q.shape != (self._state_space_size, self._action_space_size):
            raise ValueError(
                f"Loaded Q-table shape {q.shape} does not match expected "
                f"({self._state_space_size}, {self._action_space_size})"
            )
        # Ensure dtype consistency
        self._q_table = q.astype(np.float64, copy=False)

    # -----------------
    # Internal helpers
    # -----------------
    def _validate_state(self, state: int) -> None:
        if not (0 <= int(state) < self._state_space_size):
            raise IndexError(f"state {state} out of bounds [0, {self._state_space_size})")

    def _validate_action(self, action: int) -> None:
        if not (0 <= int(action) < self._action_space_size):
            raise IndexError(f"action {action} out of bounds [0, {self._action_space_size})")