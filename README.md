# 🏭 AI-Driven Digital Twin for Manufacturing Optimization

This repository contains the full implementation of a research framework that compares traditional manufacturing systems with an AI-enhanced system using Digital Twins, Multi-Agent Reinforcement Learning (MARL), and Physics-Informed AI models.

## 📌 Project Overview

Manufacturing systems often suffer from latency, static workflows, and inefficient decision-making. This project proposes an AI-powered orchestration framework that dramatically improves:

- Real-time responsiveness (reduced latency from 12.7s to <500ms)
- Quality control (defect rate reduced from 6.2% to 1.1%)
- Overall Equipment Effectiveness (OEE improved from 82.4% to 94.1%)
- Workflow reconfiguration time (3–5 hours to 27 seconds)

## 🔍 Key Features

- ✅ Physics-Informed Digital Twin Synchronization  
- 🤖 Multi-Agent Reinforcement Learning (RLlib + PPO)  
- 📊 Real-time KPI Dashboard for OEE, Quality, Energy, Maintenance  
- 🔁 Generative AI for anomaly scenario testing  
- 🧠 Agentic AI system using language models for high-level reasoning  
- 🔍 Validation with MAE, DTW, and Sensitivity Analysis  
- 📉 Comparative simulation of baseline vs enhanced system  

## 📂 Directory Structure

```
├── main.py                              # Main entry point
├── requirements.txt                     # Python dependencies
├── configs/                             # Configuration files
│   └── config.json                      # System configuration
├── environment/                         # Enhanced MARL environment
│   └── manufacturing_env.py             # Manufacturing environment implementation
├── orchestrator/                        # AI orchestrator logic
│   └── orchestrator.py                  # Orchestrator implementation
├── models/                              # Custom RL and LLM-based agent models
│   ├── digital_twin.py                  # Digital twin implementation
│   └── marl_agents.py                   # MARL agents implementation
├── validation/                          # Evaluation metrics and validation tools
│   └── kpi_dashboard.py                 # KPI dashboard and goal planning
├── notebooks/                           # Original Jupyter notebooks
│   ├── MARL.ipynb                       # Main AI-enhanced framework
│   └── Traditional_Baseline.ipynb       # SimPy-based traditional process model
├── plots/                               # Timeline and KPI comparison charts
├── tests/                               # Unit and integration tests
└── README.md                            # Project documentation
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**:
   ```bash
   python main.py
   ```

3. **Run Tests**:
   ```bash
   python -m pytest tests/
   ```

## 🧪 Key Components

### Digital Twin
The digital twin component synchronizes with the real manufacturing system and provides a physics-informed simulation environment.

### Multi-Agent Reinforcement Learning
The MARL system uses RLlib with PPO algorithms to optimize manufacturing decisions across multiple agents.

### Orchestrator
The central orchestrator coordinates between the digital twin, MARL agents, and KPI dashboard to make optimal decisions.

### KPI Dashboard
Tracks key performance indicators and provides real-time feedback on system performance.

## 📈 Performance Improvements

- **Latency Reduction**: From 12.7s to <500ms
- **Quality Improvement**: Defect rate reduced from 6.2% to 1.1%
- **OEE Enhancement**: Improved from 82.4% to 94.1%
- **Reconfiguration Time**: Reduced from 3–5 hours to 27 seconds

## 📚 Research Documentation

The repository includes a comprehensive PDF document detailing the Quantum Crew framework for real-time dynamic optimization.

## 🛠️ Development

This project follows a modular architecture with clear separation of concerns. Each component can be developed and tested independently.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.