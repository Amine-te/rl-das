# RL-DAS Experiments Guide

**Reinforcement Learning for Dynamic Algorithm Selection on TSP**

This directory contains the complete pipeline for training and evaluating RL agents that dynamically select optimization algorithms (GA, TS, SA, ILS) to solve Traveling Salesperson Problems.

---

## 🚀 Quick Start Workflow

### 1. Install Dependencies
```bash
# From project root
pip install -r requirements.txt
cd experiments
```

### 2. Train an Agent
```bash
# Train for 1M timesteps, save to ./my_experiment
python train.py --timesteps 1000000 --checkpoint-dir ./my_experiment
```

### 3. Monitor Progress
```bash
# Open http://localhost:6006
tensorboard --logdir ./my_experiment/logs
```

### 4. Evaluate & Compare
```bash
# Compare trained agent against single-algorithm baselines
python evaluate.py \
  --model ./my_experiment/best_model.zip \
  --num-test-instances 50 \
  --run-baselines
```

---

## 📂 Directory Structure

When running experiments, the system organizes outputs automatically:

```
experiments/
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
│
└── my_experiment/            # Created by --checkpoint-dir
    ├── best_model.zip        # ★ Load this for evaluation!
    ├── checkpoints/          # Regular snapshots
    │   └── run_checkpoint_50000_steps.zip
    └── logs/                 # TensorBoard metrics
```

---

## 🏋️ Algorithm Portfolio

The agent selects from 4 complementary algorithms to solve the problem:

| Algorithm (Action) | Role | Strength |
|-------------------|------|----------|
| **0: GA** (Genetic) | Exploration | Population diversity, finding good regions |
| **1: TS** (Tabu Search) | Intensification | Systematic local search, memory-based |
| **2: SA** (Simulated Annealing) | Balance | Probabilistic acceptance, escaping local optima |
| **3: ILS** (Iterated HS) | Perturbation | Modification + refinement to jump basins |

---

## 🛠️ Detailed Usage

### Training (`train.py`)

Trains the PPO agent.

**Key Arguments:**
- `--timesteps`: Total steps (Default: 1M). Use 100K for dry runs.
- `--checkpoint-dir`: **Important**. Where to save everything.
- `--num-cities`: Problem size (Default: 50).
- `--num-envs`: Parallel environments (Default: 4). Increase for speed.

**Resource Guide:**
| Hardware | Settings | Est. Time (1M steps) |
|----------|----------|----------------------|
| Laptop (4 core) | `--num-envs 4` | ~2-3 hours |
| Server (16 core) | `--num-envs 16` | ~30-45 mins |
| Debugging | `--num-envs 1` | ~30 mins (100K) |

**Example: Robust Training**
```bash
python train.py \
  --timesteps 1000000 \
  --num-cities 50 \
  --instance-type mixed \
  --checkpoint-dir ./exp_mixed_50
```

### Evaluation (`evaluate.py`)

Tests the agent and provides detailed analysis.

**Key Arguments:**
- `--model`: Path to `.zip` file (usually `best_model.zip`).
- `--num-test-instances`: How many problems to solve (Default: 20).
- `--run-baselines`: **Highly Recommended**. Runs GA, TS, SA, ILS standalone to prove RL superiority.
- `--verbose`: Shows per-instance details.

**Interpreting Output:**
The script generates a report in `results/` similar to:
```
RL AGENT PERFORMANCE SUMMARY
Mean Cost:   123.456 ± 5.678
Improvement over baselines:
  vs GA: +5.12%   (Positive = RL is better)
  vs TS: +2.01%
```

**Step-by-Step Logs:**
Look at the generated `.txt` file to see the agent's decisions:
```
Step 0: GA (Exp) -> Cost 220.3
Step 1: TS (Int) -> Cost 210.1
Step 2: SA (Bal) -> Cost 205.6
```

---

## ❓ Troubleshooting

**Q: Training is running out of memory.**
A: Reduce `--num-envs` (e.g., to 2) or `--population-size`.

**Q: Evaluation fails with "Observation space mismatch".**
A: Ensure `--num-cities` in evaluation matches what was used in training.

**Q: Agent isn't beating baselines.**
A: 
1. Train longer (`--timesteps`).
2. Use `mixed` instance type for better generalization.
3. Increase `--ent-coef` (0.01 -> 0.05) to force more exploration.

**Q: How do I resume training?**
A: `python train.py --resume ./my_exp/checkpoints/run_checkpoint_100000_steps.zip --checkpoint-dir ./my_exp`
