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

---

## 📂 Directory Structure

```bash
├── baseline_traditional_simulation.ipynb   # SimPy-based traditional process model
├── digital_twin_marl_system.ipynb         # Main AI-enhanced framework
├── environment/                           # Enhanced MARL environment
├── orchestrator/                          # AI orchestrator logic
├── models/                                # Custom RL and LLM-based agent models
├── validation/                            # Evaluation metrics and validation tools
├── plots/                                 # Timeline and KPI comparison charts
└── README.md                              # Project documentation
