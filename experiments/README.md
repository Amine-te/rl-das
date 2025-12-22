# RL-DAS Experiments

## 🚀 Quick Start

All training and evaluation logic is now unified into `train.py` and `evaluate.py`.

### 1. Train Agent

**Option A: Train DQN on TSPLIB (Recommended)**
Uses real benchmark instances with data augmentation (8x).

```bash
python train.py --model-type dqn --tsplib-dir data/tsplib --timesteps 500000 --buffer-size 100000 --checkpoint-dir checkpoints/dqn_tsplib
```

**Option B: Train PPO on Synthetic Data**
Uses generated random instances.

```bash
python train.py --model-type ppo --timesteps 500000 --num-instances 200 --ent-coef 0.02 --checkpoint-dir checkpoints/ppo_synthetic
```

### 2. Evaluate

Compare against single-algorithm baselines on synthetic or real data.

**Evaluate on Training Data (TSPLIB):**

```bash
python evaluate.py --model checkpoints/dqn_tsplib/best_model.zip --tsplib-dir data/tsplib --run-baselines
```

**Evaluate on New Synthetic Data:**

```bash
python evaluate.py --model checkpoints/dqn_tsplib/best_model.zip --num-cities 50 --num-test-instances 20 --run-baselines
```

## 🔄 Resuming Training

To continue training from a saved checkpoint:

```bash
python train.py \
  --resume checkpoints/dqn_run1/checkpoints/dqn_run1_checkpoint_100000_steps.zip \
  --timesteps 500000 \
  --checkpoint-dir checkpoints/dqn_run1_continued
```

## ⚙️ Common Configurations

| Feature           | Flag                                                 |
| :---------------- | :--------------------------------------------------- |
| **Model Type**    | `--model-type [dqn, ppo]`                            |
| **Real Data**     | `--tsplib-dir data/tsplib`                           |
| **Augmentation**  | `--augment` (Default True for TSPLIB)                |
| **Normalization** | `--normalize` (Default True)                         |
| **Fixed Size**    | `--num-cities 50`                                    |
| **Explore More**  | `--initial-eps 1.0` (DQN) or `--ent-coef 0.05` (PPO) |

## 📊 Outputs

- **Models**: `checkpoints/RunName/best_model.zip`
- **Logs**: `tensorboard --logdir checkpoints/RunName/logs`
- **Results**: `results/RunName.txt`
