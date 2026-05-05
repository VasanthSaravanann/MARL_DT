#!/usr/bin/env python3
"""
Test runner for the AI-Driven Digital Twin for Manufacturing Optimization
"""
import subprocess
import sys
import os

def run_tests():
    """Run all tests in the tests directory."""
    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Run tests
    try:
        # Run environment tests
        print("Running environment tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_environment.py", "-v"
        ], cwd=project_root, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"Environment tests failed with return code {result.returncode}")
            return False

        # Run MARL agents tests
        print("\nRunning MARL agents tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_marl_agents.py", "-v"
        ], cwd=project_root, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"MARL agents tests failed with return code {result.returncode}")
            return False

        print("\nAll tests passed!")
        return True

    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)