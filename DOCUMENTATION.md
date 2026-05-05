# AI-Driven Digital Twin for Manufacturing Optimization

## Overview

This project implements an AI-driven digital twin system for manufacturing optimization using Multi-Agent Reinforcement Learning (MARL). The system consists of several interconnected components that work together to optimize manufacturing processes in real-time.

## Components

### 1. Environment (`environment/manufacturing_env.py`)

The manufacturing environment simulates a factory floor with multiple agents (machines and robots) that interact with each other. It extends PettingZoo's ParallelEnv to provide a standardized interface for multi-agent reinforcement learning.

#### Key Features:
- Physics-based state transitions for realistic manufacturing simulation
- Multiple agents representing machines and robots
- Four-dimensional observation space: [throughput, quality, energy, maintenance]
- Discrete action space with four actions: idle, produce, maintain, adjust_quality
- Reward calculation based on throughput, quality, energy efficiency, and maintenance

### 2. Models (`models/`)

#### Digital Twin (`models/digital_twin.py`)
The digital twin component maintains a virtual representation of the physical manufacturing system. It includes:

- PhysicsInformedDigitalTwin: A physics-informed model that mirrors the real system
- DigitalTwinSynchronizer: Synchronizes the digital twin with real-world data
- EnhancedSynchronizer: Advanced synchronization with generative model capabilities

#### MARL Agents (`models/marl_agents_simple.py`)
Multi-Agent Reinforcement Learning implementation optimized for manufacturing systems:

- SimpleActionMaskModel: Action masking for valid operations
- PhysicsConstraintLayer: Enforces physics constraints on generated data
- PhysicsInformedDigitalTwin: Digital twin model for MARL
- MARLAgentSystem: Manages multiple MARL agents
- CoordinatedMARLSystem: Coordinates multiple agents
- MixingNetwork: Combines individual agent values into joint values

### 3. Orchestrator (`orchestrator/orchestrator.py`)

The orchestrator coordinates all AI components and makes high-level decisions:

- AIOrchestrator: Basic coordination between components
- AgenticOrchestrator: High-level reasoning and decision-making
- EnhancedAIOrchestrator: Integrates multiple AI paradigms

### 4. Validation (`validation/kpi_dashboard.py`)

Validation components for monitoring and evaluating system performance:

- KPIDashboard: Tracks key performance indicators
- ManufacturingGoalPlanner: Plans manufacturing goals based on KPI targets

## Configuration

Configuration is managed through `configs/config.json` which includes settings for:
- Environment parameters
- Digital twin synchronization
- MARL algorithm parameters
- KPI targets
- Training parameters

## Usage

Run the system with:
```bash
python main.py --episodes 10
```

Additional parameters:
- `--config`: Path to configuration file (default: configs/config.json)
- `--episodes`: Number of episodes to run (default: 10)

## Dependencies

Core dependencies:
- numpy: Numerical computing
- torch: PyTorch for neural networks
- gymnasium: Reinforcement learning environments
- pettingzoo: Multi-agent reinforcement learning framework
- simpy: Discrete-event simulation

Development dependencies:
- pytest: Testing framework
- black: Code formatting
- flake8: Code linting