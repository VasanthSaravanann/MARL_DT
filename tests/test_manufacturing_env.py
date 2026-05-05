#!/usr/bin/env python3
"""
Tests for the manufacturing environment module
"""
import sys
import os
import pytest
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.manufacturing_env import EnhancedManufacturingEnv


def test_environment_initialization():
    """Test that the environment initializes correctly."""
    env = EnhancedManufacturingEnv()

    # Check that agents are properly initialized
    assert len(env.possible_agents) == 3
    assert "machine_1" in env.possible_agents
    assert "machine_2" in env.possible_agents
    assert "robot_1" in env.possible_agents

    # Check observation and action spaces
    for agent in env.possible_agents:
        assert agent in env.observation_spaces
        assert agent in env.action_spaces
        assert env.observation_spaces[agent].shape == (4,)
        assert env.action_spaces[agent].n == 4


def test_environment_reset():
    """Test that the environment resets correctly."""
    env = EnhancedManufacturingEnv()
    observations, infos = env.reset()

    # Check that observations are returned for all agents
    assert len(observations) == 3
    for agent in env.possible_agents:
        assert agent in observations
        assert len(observations[agent]) == 4
        # Check that values are in valid range [0, 1]
        assert all(0 <= val <= 1 for val in observations[agent])


def test_environment_step():
    """Test that the environment steps correctly."""
    env = EnhancedManufacturingEnv()
    observations, infos = env.reset()

    # Create dummy actions for all agents
    actions = {agent: 1 for agent in env.possible_agents}

    # Step the environment
    next_observations, rewards, terminations, truncations, infos = env.step(actions)

    # Check that all return values have the correct structure
    assert len(next_observations) == 3
    assert len(rewards) == 3
    assert len(terminations) == 3
    assert len(truncations) == 3
    assert len(infos) == 3

    # Check that observations are updated
    for agent in env.possible_agents:
        assert agent in next_observations
        assert len(next_observations[agent]) == 4


def test_physics_step():
    """Test the physics-based state transitions."""
    env = EnhancedManufacturingEnv()

    # Test produce action (action 1)
    current_state = np.array([0.5, 0.8, 0.3, 0.7], dtype=np.float32)
    new_state = env._physics_step("machine_1", 1, current_state)

    # Check that state is updated appropriately
    assert len(new_state) == 4
    assert all(0 <= val <= 1 for val in new_state)

    # Test maintain action (action 2)
    new_state = env._physics_step("machine_1", 2, current_state)

    # Check that maintenance level increases
    assert new_state[3] >= current_state[3]
    assert all(0 <= val <= 1 for val in new_state)


if __name__ == "__main__":
    pytest.main([__file__])