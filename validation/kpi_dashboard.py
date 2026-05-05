#!/usr/bin/env python3
"""
KPI Dashboard for Manufacturing Systems
"""
import numpy as np
from typing import Dict, Any


class KPIDashboard:
    """
    Dashboard for tracking and calculating manufacturing KPIs.
    """

    def __init__(self):
        """
        Initialize the KPI dashboard.
        """
        self.kpis = {
            "throughput": [],
            "quality": [],
            "energy_consumption": [],
            "maintenance_cost": [],
            "oee": []
        }

    def update(self, environment_state, agent_actions):
        """
        Update KPIs based on current environment state and agent actions.

        Args:
            environment_state: Current state of the environment
            agent_actions: Actions taken by agents

        Returns:
            dict: Current KPIs
        """
        # Calculate KPIs based on current state and actions
        current_kpis = self._calculate_kpis(environment_state, agent_actions)

        # Update KPI history
        for kpi_name, value in current_kpis.items():
            if kpi_name in self.kpis:
                self.kpis[kpi_name].append(value)

        # Return current KPIs for display or logging
        return current_kpis

    def _calculate_kpis(self, state, actions):
        """
        Calculate KPIs based on current state and actions.

        Args:
            state: Current state of the environment
            actions: Actions taken by agents

        Returns:
            dict: Calculated KPIs
        """
        # Handle different state formats
        if isinstance(state, dict):
            # State is a dictionary with agent states
            agent_states = list(state.values())
            throughput = sum(s[0] if len(s) > 0 else 0 for s in agent_states)
            quality = sum(s[1] if len(s) > 1 else 0 for s in agent_states) / len(agent_states) if agent_states else 0
            energy = sum(s[2] if len(s) > 2 else 0 for s in agent_states)

            # Count maintenance actions
            maintenance = sum(1 for action in actions.values() if action == 2) if hasattr(actions, 'values') else 0

            # Calculate availability based on maintenance levels
            availability = sum(s[3] if len(s) > 3 else 0 for s in agent_states) / len(agent_states) if agent_states else 0
        else:
            # Assume state is a single array
            throughput = state[0] if len(state) > 0 else 0
            quality = state[1] if len(state) > 1 else 0
            energy = state[2] if len(state) > 2 else 0
            maintenance = 0  # Default if no actions provided
            availability = state[3] if len(state) > 3 else 0

        # To avoid division by zero, set a theoretical max throughput
        theoretical_max = 100.0  # <-- you can modify this as needed
        performance = throughput / theoretical_max if theoretical_max != 0 else 0.0

        oee = availability * performance * quality if availability > 0 and performance > 0 and quality > 0 else 0.0

        return {
            "throughput": throughput,
            "quality": quality,
            "energy_consumption": energy,
            "maintenance_cost": maintenance * 0.5,
            "oee": oee
        }


class ManufacturingGoalPlanner:
    """
    Planner for setting manufacturing goals based on KPIs.
    """

    def __init__(self, kpi_dashboard, digital_twin):
        """
        Initialize the manufacturing goal planner.

        Args:
            kpi_dashboard: KPI dashboard component
            digital_twin: Digital twin component
        """
        self.kpi_dashboard = kpi_dashboard
        self.digital_twin = digital_twin
        self.goal_hierarchy = {
            "primary": ["maximize_throughput", "minimize_defects", "reduce_energy"],
            "secondary": ["optimize_inventory", "balance_workload", "reduce_waste"]
        }

    def set_manufacturing_goals(self, current_state, target_kpis):
        """
        Set goals based on KPI gaps and manufacturing state.

        Args:
            current_state: Current state of the system
            target_kpis: Target KPIs to achieve

        Returns:
            list: Prioritized goals
        """
        # Calculate current KPIs
        current_kpis = self.kpi_dashboard._calculate_kpis(current_state, {})

        # Calculate KPI gaps
        kpi_gaps = self._calculate_kpi_gaps(current_kpis, target_kpis)

        # Prioritize goals based on gaps
        prioritized_goals = self._prioritize_goals(kpi_gaps)

        return prioritized_goals

    def _calculate_kpi_gaps(self, current_kpis, target_kpis):
        """
        Calculate gaps between current and target KPIs.

        Args:
            current_kpis: Current KPIs
            target_kpis: Target KPIs

        Returns:
            dict: KPI gaps
        """
        gaps = {}
        for kpi, current_value in current_kpis.items():
            if kpi in target_kpis:
                # For KPIs where higher is better (like throughput)
                if kpi in ["throughput"]:
                    gaps[kpi] = (target_kpis[kpi] - current_value) / target_kpis[kpi]
                # For KPIs where lower is better (like energy, maintenance_cost)
                else:
                    gaps[kpi] = (current_value - target_kpis[kpi]) / target_kpis[kpi]

        return gaps

    def _prioritize_goals(self, kpi_gaps):
        """
        Prioritize goals based on KPI gaps.

        Args:
            kpi_gaps: KPI gaps

        Returns:
            list: Prioritized goals
        """
        # Sort gaps by magnitude
        sorted_gaps = sorted(kpi_gaps.items(), key=lambda x: abs(x[1]), reverse=True)

        # Map KPIs to goals
        kpi_to_goal = {
            "throughput": "maximize_throughput",
            "quality": "minimize_defects",
            "energy_consumption": "reduce_energy",
            "maintenance_cost": "optimize_maintenance"
        }

        # Create prioritized goal list
        prioritized_goals = []
        for kpi, gap in sorted_gaps:
            if kpi in kpi_to_goal:
                prioritized_goals.append({
                    "goal": kpi_to_goal[kpi],
                    "gap": gap,
                    "priority": "high" if abs(gap) > 0.2 else "medium" if abs(gap) > 0.1 else "low"
                })

        return prioritized_goals