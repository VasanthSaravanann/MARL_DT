#!/usr/bin/env python3
"""
Simple test script to check if the environment works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.manufacturing_env import EnhancedManufacturingEnv

def test_environment():
    """Test the environment initialization and basic functionality."""
    print("Testing environment initialization...")

    # Create environment
    env = EnhancedManufacturingEnv()
    print("Environment created successfully!")

    # Test reset
    observations, infos = env.reset()
    print("Environment reset successfully!")
    print(f"Observations: {observations}")

    # Test step
    actions = {agent: 1 for agent in env.agents}  # All agents produce
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("Environment step executed successfully!")
    print(f"Rewards: {rewards}")

    # Render
    env.render()

    print("All tests passed!")

if __name__ == "__main__":
    test_environment()