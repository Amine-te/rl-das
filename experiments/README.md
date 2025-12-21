# RL-DAS Experiments

## 🚀 Quick Start

**1. Train Agent (Recommended)**
Trains on variable 30-75 city instances for robustness:

```bash
python train.py --timesteps 500000 --num-instances 200 --ent-coef 0.1 --checkpoint-dir ../checkpoints/run1
```

**2. Evaluate**
Compare against single-algorithm baselines:

```bash
python evaluate.py --model ../checkpoints/run1/best_model.zip --num-test-instances 20 --run-baselines --deterministic
```

## 🔄 Resuming Training

To continue training from a saved checkpoint:

```bash
python train.py \
  --resume ../checkpoints/run1/checkpoints/run1_checkpoint_100000_steps.zip \
  --timesteps 500000 \
  --checkpoint-dir ../checkpoints/run1_continued
```

## ⚙️ Common Configurations

| Scenario                    | Command Flags                                            |
| --------------------------- | -------------------------------------------------------- |
| **Variable Size** (Default) | _(No size flags required)_                               |
| **Fixed Size** (e.g. 50)    | `--num-cities 50`                                        |
| **Production Run**          | `--timesteps 1000000 --num-instances 500 --ent-coef 0.1` |
| **Debug Run**               | `--timesteps 50000 --num-envs 1`                         |

## � Outputs

- **Models**: `../checkpoints/run1/best_model.zip`
- **Logs**: `tensorboard --logdir ../checkpoints/run1/logs`
- **Results**: `results/eval_*.txt`
