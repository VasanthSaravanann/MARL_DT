#!/usr/bin/env python3
"""
Test suite for MARL agents in the AI-Driven Digital Twin for Manufacturing Optimization
"""
import sys
import os
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.marl_agents_simple import (
    SimpleActionMaskModel,
    PhysicsInformedDigitalTwin,
    MARLAgentSystem,
    PhysicsConstraintLayer
)


class TestSimpleActionMaskModel(unittest.TestCase):
    """Test cases for the simple action mask model."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.obs_space_dim = 4
        self.action_space_dim = 4
        self.model = SimpleActionMaskModel(self.obs_space_dim, self.action_space_dim)

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, SimpleActionMaskModel)

    def test_forward_pass(self):
        """Test forward pass through the model."""
        # Create test input
        batch_size = 2
        test_input = torch.randn(batch_size, self.obs_space_dim)

        # Forward pass
        output = self.model.forward(test_input)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.action_space_dim))


class TestPhysicsInformedDigitalTwin(unittest.TestCase):
    """Test cases for the physics-informed digital twin."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.obs_space_dim = 4
        self.action_space_dim = 4
        self.model = PhysicsInformedDigitalTwin(self.obs_space_dim, self.action_space_dim)

    def test_digital_twin_initialization(self):
        """Test that the digital twin initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, PhysicsInformedDigitalTwin)

    def test_digital_twin_forward(self):
        """Test forward pass through the digital twin."""
        # Create test input
        batch_size = 2
        test_input = torch.randn(batch_size, self.obs_space_dim)

        # Forward pass
        output = self.model.forward(test_input)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.action_space_dim))


class TestMARLAgentSystem(unittest.TestCase):
    """Test cases for the MARL agent system."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        digital_twin = None  # For testing, we don't need a real digital twin
        self.agent_system = MARLAgentSystem(digital_twin)

    def test_agent_system_initialization(self):
        """Test that the agent system initializes correctly."""
        self.assertIsNotNone(self.agent_system)
        self.assertIsInstance(self.agent_system, MARLAgentSystem)

    def test_compute_actions(self):
        """Test action computation."""
        # Create test observations
        test_observations = {
            "machine_1": np.array([0.5, 0.8, 0.3, 0.9]),  # High maintenance
            "machine_2": np.array([0.5, 0.8, 0.3, 0.2]),  # Low maintenance
            "robot_1": np.array([0.5, 0.8, 0.3, 0.5])     # Medium maintenance
        }

        # Compute actions
        actions = self.agent_system.compute_actions(test_observations)

        # Check that all agents have actions
        self.assertIn("machine_1", actions)
        self.assertIn("machine_2", actions)
        self.assertIn("robot_1", actions)

        # Check action types
        for action in actions.values():
            self.assertIsInstance(action, int)
            self.assertIn(action, [0, 1, 2])  # Idle, produce, maintain


class TestPhysicsConstraintLayer(unittest.TestCase):
    """Test cases for the physics constraint layer."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.constraint_layer = PhysicsConstraintLayer()

    def test_constraint_layer_initialization(self):
        """Test that the constraint layer initializes correctly."""
        self.assertIsNotNone(self.constraint_layer)
        self.assertIsInstance(self.constraint_layer, PhysicsConstraintLayer)

    def test_temperature_constraints(self):
        """Test temperature constraints."""
        # Create test input with out-of-range temperatures
        test_input = torch.tensor([150.0, 0.5, 0.5, 0.5, 150.0, 0.5, 0.5, 0.5, 150.0, 0.5, 0.5, 0.5, 0.5])

        # Apply constraints
        constrained_output = self.constraint_layer.forward(test_input)

        # Check that temperatures are clamped
        self.assertLessEqual(constrained_output[0], 100.0)
        self.assertGreaterEqual(constrained_output[0], 20.0)
        self.assertLessEqual(constrained_output[4], 100.0)
        self.assertGreaterEqual(constrained_output[4], 20.0)
        self.assertLessEqual(constrained_output[8], 100.0)
        self.assertGreaterEqual(constrained_output[8], 20.0)


if __name__ == "__main__":
    unittest.main()