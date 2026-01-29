# Sandbox: Batch Selection for RL Alignment

Research codebase for exploring lightweight batch selection methods,
with a focus on **Reinforcement Learning from Human Feedback (RLHF)**
and **Direct Preference Optimization (DPO)**.

## Research Goal

Existing batch selection methods (GREATS, RHO-Loss, GradNorm) add computational
overhead that often negates efficiency gains. This research explores:

1. **Lightweight batch selection** with <5% overhead
2. **RL-specific selection criteria** (advantages, preference margins, KL)
3. **Fair benchmarking** following "No Train No Gain" methodology

## Structure

```
Sandbox/
├── batch_selection/       # Selection strategies
│   ├── base.py           # Base class and interface
│   ├── random_selector.py
│   ├── loss_based.py     # Loss-based selection (cached/current)
│   ├── gradient_norm.py  # Gradient norm based
│   ├── uncertainty.py    # Entropy/confidence based (CHEAP!)
│   ├── reducible_loss.py # RHO-Loss style
│   └── rl_selectors.py   # Novel RL-specific selectors
│
├── models/               # Model utilities
│   └── language_models.py
│
├── datasets/             # Data loading
│   └── loaders.py        # Preference & SFT datasets
│
├── training/             # Training loops
│   ├── supervised.py     # SFT baseline
│   ├── dpo.py           # DPO training
│   └── ppo.py           # PPO/RLHF training
│
├── utils/               # Utilities
│   ├── timing.py        # Overhead measurement
│   ├── metrics.py       # Logging and comparison
│   └── visualization.py # Plotting
│
└── experiments/         # Experiment scripts
    ├── run_dpo_comparison.py
    ├── run_ppo_comparison.py
    └── run_all.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick experiment
python -m Sandbox.experiments.run_all --quick

# Run full experiment
python -m Sandbox.experiments.run_all
```

## Batch Selectors

| Selector | Overhead | Best For | Description |
|----------|----------|----------|-------------|
| `RandomSelector` | ~0% | Baseline | Standard random sampling |
| `LossBasedSelector` (cached) | ~0% | SFT | Uses cached per-sample losses |
| `LossBasedSelector` (current) | O(forward) | When fresh losses needed |
| `UncertaintySelector` | ~0%* | All | Uses prediction entropy |
| `GradientNormSelector` (last_layer) | O(forward) | Research | Last-layer grad proxy |
| `ReducibleLossSelector` | O(lookup) | All | RHO-loss style |
| `PreferenceMarginSelector` | ~0%* | DPO | Preference margin based |
| `AdvantageVarianceSelector` | ~0%* | PPO | Advantage variance |
| `CombinedRLSelector` | ~0%* | RL | Weighted combination |

*When reusing computation from training forward pass

## Novel Contribution: RL-Specific Selectors

The `rl_selectors.py` module contains novel selection strategies designed
specifically for RL alignment:

### PreferenceMarginSelector
Prioritizes samples with moderate preference margins (not too easy, not too hard).

### AdvantageVarianceSelector
Tracks advantage estimate variance; high variance indicates learning opportunity.

### KLDivergenceSelector
Selects based on policy-reference divergence.

### CombinedRLSelector
Weighted combination of multiple RL-specific signals.

## Key Design Principles

1. **Overhead Awareness**: All selectors track their overhead
2. **Fair Comparison**: Use wall-clock time, not just epochs
3. **Modular Design**: Easy to add new selectors
4. **RL Focus**: Designed for preference learning / RLHF setting

## Benchmarking Protocol

Following "No Train No Gain" (Kaddour et al., 2023):

1. Fixed compute budget (wall-clock time)
2. Track selection overhead explicitly
3. Report net speedup = (baseline_time / selector_time) × (1 - overhead_ratio)
4. Success criterion: net_speedup > 1.0 with loss degradation < 5%

## References

- GREATS (NeurIPS 2024): Online selection using Taylor expansion
- No Train No Gain (2023): Benchmarking efficient training methods
- RHO-Loss (2022): Reducible holdout loss selection
- DPO (2023): Direct Preference Optimization
