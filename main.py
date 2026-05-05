#!/usr/bin/env python3
"""
Main entry point for the AI-Driven Digital Twin for Manufacturing Optimization
"""
import os
import sys
import json
import argparse
from environment.manufacturing_env import EnhancedManufacturingEnv
from models.digital_twin import PhysicsInformedDigitalTwin, EnhancedSynchronizer
from models.marl_agents_simple import MARLAgentSystem, PhysicsInformedDigitalTwin as MARLModel
from orchestrator.orchestrator import EnhancedAIOrchestrator
from validation.kpi_dashboard import KPIDashboard, ManufacturingGoalPlanner


def load_config(config_path="configs/config.json"):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def initialize_system(config):
    """
    Initialize the manufacturing AI system.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Dictionary containing all system components
    """
    # Create environment
    env = EnhancedManufacturingEnv(config.get("environment", {}))

    # Create digital twin
    digital_twin = PhysicsInformedDigitalTwin(
        env.observation_spaces["machine_1"],
        env.action_spaces["machine_1"],
        config["environment"].get("observation_space_dim", 4)
    )

    # Create MARL agents
    marl_agents = MARLAgentSystem(digital_twin)

    # Create KPI dashboard
    kpi_dashboard = KPIDashboard()

    # Create orchestrator
    orchestrator = EnhancedAIOrchestrator(digital_twin, marl_agents, kpi_dashboard)

    return {
        "environment": env,
        "digital_twin": digital_twin,
        "marl_agents": marl_agents,
        "kpi_dashboard": kpi_dashboard,
        "orchestrator": orchestrator,
        "config": config
    }


def run_simulation(system, num_episodes=10):
    """
    Run a simulation with the manufacturing AI system.

    Args:
        system (dict): Dictionary containing all system components
        num_episodes (int): Number of episodes to run
    """
    env = system["environment"]
    orchestrator = system["orchestrator"]

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        # Reset environment
        observations, infos = env.reset()

        # Run episode
        total_reward = 0
        step_count = 0

        while step_count < 100:  # Limit episode length
            # Process state with orchestrator
            actions = orchestrator.process_manufacturing_state(observations)

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Accumulate rewards
            total_reward += sum(rewards.values())

            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                break

            step_count += 1

        print(f"Episode {episode + 1} completed with total reward: {total_reward:.2f}")
        env.render()


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description="AI-Driven Digital Twin for Manufacturing Optimization")
    parser.add_argument("--config", default="configs/config.json", help="Path to configuration file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize system
    system = initialize_system(config)

    # Run simulation
    run_simulation(system, args.episodes)

    print("\nSimulation completed!")


if __name__ == "__main__":
    main()