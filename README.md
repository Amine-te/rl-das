# 🧠 RL-DAS: Dynamic Algorithm Selection for TSP

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

> **Adaptive Intelligence for Combinatorial Optimization**  
> An advanced Reinforcement Learning framework that dynamically selects the optimal metaheuristic algorithm at runtime to solve the Traveling Salesperson Problem (TSP).

---

## 📖 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
  - [State Representation](#-state-representation)
  - [Reward Function](#-reward-function)
- [Algorithm Pool](#-algorithm-pool)
- [Experiments & Evaluation](#-experiments--evaluation)
- [Installation & Usage](#-installation--usage)
- [Dashboard](#-interactive-dashboard)

---

## 🎯 Overview

The **Traveling Salesperson Problem (TSP)** is a classic NP-hard problem. While many metaheuristics exist (Genetic Algorithms, Tabu Search, etc.), no single algorithm performs best across all problem stages and instances.

**RL-DAS** (Reinforcement Learning - Dynamic Algorithm Selection) solves this by training an agent to act as a "conductor", switching between algorithms in real-time based on the current search state.

### Key Features

- **Dynamic Switching**: Instantly adapts to stagnation or changing landscape properties.
- **Hybrid Intelligence**: Combines global exploration (GA, SA) with aggressive local search (Tabu, ILS).
- **State-of-the-Art RL**: Uses PPO/DQN agents trained with Stable Baselines 3.
- **Visual Analytics**: Full-featured Streamlit dashboard for interpreting agent decisions.

---

## 🧠 How It Works

The agent observes the optimization process and selects one of 4 algorithms to run for a fixed number of function evaluations (interval FEs).

### 📡 State Representation

The agent makes decisions based on a compact **18-dimensional state vector** (`core/state_extractor.py`):

| Feature Group          | Dim | Description                                                                     |
| :--------------------- | :-: | :------------------------------------------------------------------------------ |
| **Landscape Analysis** |  5  | Cost, Budget Consumed, Population Diversity, Improvement Potential, Convergence |
| **Status**             |  1  | `Last Improved` flag                                                            |
| **Context**            |  4  | Current Algorithm (One-Hot)                                                     |
| **History**            |  4  | Success Rates per Algorithm                                                     |
| **Freshness**          |  4  | Time since last use for each algorithm                                          |

### 💎 Reward Function

The reward system (`core/reward_calculator.py`) is designed to encourage efficiency and prevent stagnation:

- **Stick with Winner**: `+0.7` to `+1.0` (plus improvement bonus) for staying with an improving algorithm.
- **Punish Stagnation**: `-1.0` for sticking with an algorithm that isn't improving.
- **Smart Switching**: Small penalties (`-0.1` to `-0.3`) for switching when stuck, incentivizing exploration of fresh or proven algorithms over random choices.

---

## 🧰 Algorithm Pool

The agent manages a portfolio of 4 complementary metaheuristics:

1.  🧬 **Genetic Algorithm (GA)**: Population-based, good for global exploration.
2.  🚫 **Tabu Search (TS)**: Aggressive local search with memory to escape optima.
3.  🔥 **Simulated Annealing (SA)**: Probabilistic acceptance of worse solutions to avoid local optima.
4.  🔄 **Iterated Local Search (ILS)**: Perturbation + Local Search for deep optimization.

---

## 🧪 Experiments & Evaluation

The project includes a robust training pipeline supporting both synthetic data and TSPLIB benchmarks.

### Training

Train the agent using PPO or DQN:

```bash
# Train PPO agent on TSPLIB
python experiments/train.py --model-type ppo --tsplib-dir data/tsplib --num-envs 4

# Train DQN on synthetic data
python experiments/train.py --model-type dqn --timesteps 100000 --instance-type clustered
```

### Evaluation

Run the evaluation script to compare the agent against baselines:

```bash
# Evaluate a trained model against baselines on 50-city problems
python experiments/evaluate.py --model checkpoints/best_model.zip --run-baselines --num-cities 50 --num-test-instances 20
```

The agent is evaluated against standalone baselines on:

- **Final Cost**: Solution quality.
- **Convergence Speed**: How fast it reaches good solutions.
- **Robustness**: Performance across different distributions (Random, Clustered, Mixed).

---

## 🚀 Installation & Usage

1.  **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/rl-das.git
    cd rl-das
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**
    ```bash
    streamlit run dashboard.py
    ```

---

## 📊 Interactive Dashboard

The project features a **Streamlit** dashboard for deep analysis:

- **Comparison Plots**: Real-time convergence curves vs. Baselines.
- **Decision Timeline**: Visualize exactly when and why the agent switched algorithms.
- **Probability Heatmaps**: See the agent's confidence (PPO) or Q-values (DQN) over time.
- **Tour Visualization**: Plot the final TSP tour on a 2D map.

---

_Built with ❤️ using PyTorch, Stable Baselines 3, and Streamlit._
