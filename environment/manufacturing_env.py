#!/usr/bin/env python3
"""
Enhanced Manufacturing Environment for Multi-Agent Reinforcement Learning
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


class EnhancedManufacturingEnv(ParallelEnv):
    """
    An enhanced manufacturing environment with physics-based parameters and
    realistic manufacturing constraints.
    """

    def __init__(self, config=None):
        """
        Initialize the manufacturing environment.

        Args:
            config (dict): Configuration parameters for the environment
        """
        self.possible_agents = ["machine_1", "machine_2", "robot_1"]
        self.agents = self.possible_agents.copy()

        # Physics-based parameters
        self.machine_params = {
            "machine_1": {"max_throughput": 0.8, "energy_factor": 1.2},
            "machine_2": {"max_throughput": 1.0, "energy_factor": 1.5},
            "robot_1": {"max_throughput": 0.6, "energy_factor": 0.8}
        }

        # Observation space: [throughput, quality, energy, maintenance]
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Action space: [idle, produce, maintain, adjust_quality]
        self.action_spaces = {
            agent: spaces.Discrete(4)
            for agent in self.possible_agents
        }

        # Initialize state
        self._agent_states = {
            agent: np.array([0.5, 0.8, 0.3, 0.7], dtype=np.float32)
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
            tuple: (observations, infos)
        """
        # Reset agent states to initial values
        self._agent_states = {
            agent: np.array([0.5, 0.8, 0.3, 0.7], dtype=np.float32)
            for agent in self.possible_agents
        }

        observations = {
            agent: self._agent_states[agent].copy()
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Execute one time step within the environment.

        Args:
            actions (dict): Actions for each agent

        Returns:
            tuple: (observations, rewards, terminations, truncations, infos)
        """
        # Apply physics-based state transitions
        for agent, action in actions.items():
            if agent in self._agent_states:
                current_state = self._agent_states[agent]
                new_state = self._physics_step(agent, action, current_state)
                self._agent_states[agent] = new_state

        # Calculate observations, rewards, and termination conditions
        observations = {
            agent: self._agent_states[agent].copy()
            for agent in self.agents
        }

        rewards = self._compute_rewards()
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _physics_step(self, agent, action, current_state):
        """
        Physics-based state transitions.

        Args:
            agent (str): Agent identifier
            action (int): Action taken by the agent
            current_state (np.array): Current state of the agent

        Returns:
            np.array: New state after applying action
        """
        params = self.machine_params[agent]
        new_state = current_state.copy()

        if action == 1:  # Produce
            new_state[0] = min(current_state[0] + 0.1, params["max_throughput"])
            new_state[2] += new_state[0] * params["energy_factor"]
            new_state[1] = max(current_state[1] - 0.02, 0.7)
            new_state[3] = max(current_state[3] - 0.05, 0)

        elif action == 2:  # Maintain
            new_state[3] = min(current_state[3] + 0.2, 1.0)

        elif action == 3:  # Adjust quality
            new_state[1] = min(current_state[1] + 0.1, 1.0)

        # Natural decay for unused states
        else:  # Idle
            new_state[0] = max(current_state[0] - 0.05, 0)
            new_state[1] = max(current_state[1] - 0.01, 0)
            new_state[2] = max(current_state[2] - 0.1, 0)
            new_state[3] = max(current_state[3] - 0.02, 0)

        return np.clip(new_state, 0, 1)

    def _compute_rewards(self):
        """
        Compute rewards for all agents based on current states.

        Returns:
            dict: Rewards for each agent
        """
        rewards = {}
        for agent, state in self._agent_states.items():
            # Reward based on throughput, quality, and energy efficiency
            throughput_reward = state[0]  # Higher throughput is better
            quality_reward = state[1]     # Higher quality is better
            energy_penalty = -0.1 * state[2]  # Lower energy consumption is better
            maintenance_bonus = 0.2 * state[3] if state[3] > 0.8 else 0  # Bonus for good maintenance

            rewards[agent] = throughput_reward + quality_reward + energy_penalty + maintenance_bonus

        return rewards

    def render(self):
        """
        Render the environment (placeholder).
        """
        print("Current states:")
        for agent, state in self._agent_states.items():
            print(f"  {agent}: Throughput={state[0]:.2f}, Quality={state[1]:.2f}, "
                  f"Energy={state[2]:.2f}, Maintenance={state[3]:.2f}")


# Register the environment
def register_env():
    """
    Register the environment with PettingZoo.
    """
    from pettingzoo.utils.to_parallel import to_parallel
    return to_parallel(EnhancedManufacturingEnv())