# RL-DAS Experiments

## 🚀 Quick Start

### 1. Train Agent (PPO or DQN)

**Option A: PPO (Recommended for continuous control / stability)**

```bash
python train_ppo.py --timesteps 500000 --num-instances 200 --ent-coef 0.05 --checkpoint-dir ../checkpoints/ppo_run1
```

**Option B: DQN (Better for discrete algorithm selection)**

```bash
python train_dqn.py --timesteps 500000 --num-instances 200 --buffer-size 100000 --checkpoint-dir ../checkpoints/dqn_run1
```

### 2. Evaluate

Compare against single-algorithm baselines:

**Evaluate PPO:**

```bash
python evaluate_ppo.py --model ../checkpoints/ppo_run1/best_model.zip --num-test-instances 20 --run-baselines --deterministic
```

**Evaluate DQN:**

```bash
python evaluate_dqn.py --model ../checkpoints/dqn_run1/best_model.zip --num-test-instances 20 --run-baselines --deterministic
```

## 🔄 Resuming Training

To continue training from a saved checkpoint:

```bash
python train_ppo.py \
  --resume ../checkpoints/run1/checkpoints/run1_checkpoint_100000_steps.zip \
  --timesteps 500000 \
  --checkpoint-dir ../checkpoints/run1_continued
```

## ⚙️ Common Configurations

| Scenario                    | Command Flags                                                  |
| :-------------------------- | :------------------------------------------------------------- |
| **Variable Size** (Default) | _(No size flags required)_                                     |
| **Fixed Size** (e.g. 50)    | `--num-cities 50`                                              |
| **Production Run**          | `--timesteps 1000000 --num-instances 500`                      |
| **Explore More**            | `--ent-coef 0.05` (PPO) or `--exploration-final-eps 0.1` (DQN) |

## 📊 Outputs

- **Models**: `../checkpoints/RunName/best_model.zip`
- **Logs**: `tensorboard --logdir ../checkpoints/RunName/logs`
- **Results**: `results/RunName.txt`
