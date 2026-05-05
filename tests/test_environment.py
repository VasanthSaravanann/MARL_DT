#!/usr/bin/env python3
"""
Test suite for the AI-Driven Digital Twin for Manufacturing Optimization
"""
import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.manufacturing_env import EnhancedManufacturingEnv


class TestManufacturingEnvironment(unittest.TestCase):
    """Test cases for the manufacturing environment."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.env = EnhancedManufacturingEnv()

    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertEqual(len(self.env.possible_agents), 3)
        self.assertIn("machine_1", self.env.possible_agents)
        self.assertIn("machine_2", self.env.possible_agents)
        self.assertIn("robot_1", self.env.possible_agents)

    def test_reset(self):
        """Test environment reset functionality."""
        observations, infos = self.env.reset()

        # Check that all agents have observations
        for agent in self.env.possible_agents:
            self.assertIn(agent, observations)
            self.assertIn(agent, infos)

        # Check observation structure
        for agent_obs in observations.values():
            self.assertEqual(len(agent_obs), 4)  # 4-dimensional observation space
            self.assertTrue(np.all(agent_obs >= 0))
            self.assertTrue(np.all(agent_obs <= 1))

    def test_step(self):
        """Test environment step functionality."""
        # Reset environment first
        observations, infos = self.env.reset()

        # Take a step with all agents producing
        actions = {agent: 1 for agent in self.env.agents}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Check that all agents have observations and rewards
        for agent in self.env.agents:
            self.assertIn(agent, observations)
            self.assertIn(agent, rewards)
            self.assertIn(agent, terminations)
            self.assertIn(agent, truncations)
            self.assertIn(agent, infos)

        # Check observation structure
        for agent_obs in observations.values():
            self.assertEqual(len(agent_obs), 4)

        # Check rewards are numeric
        for reward in rewards.values():
            self.assertIsInstance(reward, (int, float, np.floating))

    def test_physics_step(self):
        """Test physics-based state transitions."""
        # Reset environment
        observations, infos = self.env.reset()

        # Test produce action (action=1)
        actions = {agent: 1 for agent in self.env.agents}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Check that throughput increased (action=1 should increase throughput)
        for agent_obs in observations.values():
            self.assertGreaterEqual(agent_obs[0], 0.5)  # Started at 0.5

    def test_action_effects(self):
        """Test that different actions have different effects."""
        # Reset environment
        observations, infos = self.env.reset()
        initial_observations = observations.copy()

        # Test maintain action (action=2) - should increase maintenance
        actions = {agent: 2 for agent in self.env.agents}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Check that maintenance increased
        for agent in self.env.agents:
            self.assertGreaterEqual(observations[agent][3], initial_observations[agent][3])


class TestObservationAndActionSpaces(unittest.TestCase):
    """Test cases for observation and action spaces."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.env = EnhancedManufacturingEnv()

    def test_observation_spaces(self):
        """Test that observation spaces are correctly defined."""
        for agent, space in self.env.observation_spaces.items():
            self.assertEqual(space.shape[0], 4)
            self.assertEqual(space.low[0], 0)
            self.assertEqual(space.high[0], 1)

    def test_action_spaces(self):
        """Test that action spaces are correctly defined."""
        for agent, space in self.env.action_spaces.items():
            self.assertEqual(space.n, 4)  # 4 discrete actions


if __name__ == "__main__":
    unittest.main()