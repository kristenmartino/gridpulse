# ⚡ WattCast Autoresearch
### Autonomous ML Optimization for Energy Demand Forecasting

Applies [Karpathy's autoresearch pattern](https://github.com/karpathy/autoresearch) to
WattCast's weather-aware energy demand forecasting pipeline.

An AI agent iterates on `train.py` — modifying feature engineering, hyperparameters,
model architecture, and preprocessing — then evaluates each variant against MAPE on
a held-out validation set. Improvements are kept and committed to git. Regressions
are reverted. The loop runs indefinitely while you sleep.

**Baseline: 3.13% MAPE across 8 U.S. grid regions.**

---

## Quick Start

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd wattcast-autoresearch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data (one-time)
python prepare.py

# 4. Run baseline to verify setup
python train.py

# 5. Point your agent at program.md and let it rip
#    With Claude Code:
claude "Read program.md and start running experiments on train.py"
```

---

## Architecture

```
wattcast-autoresearch/
├── prepare.py          # Data download + feature store build (READ-ONLY)
├── train.py            # The editable asset — agent modifies this
├── program.md          # Research directions — human modifies this
├── requirements.txt    # Dependencies
├── README.md           # This file
├── data/               # Created by prepare.py
│   ├── features.parquet
│   ├── train.parquet
│   └── val.parquet
└── results/            # Created by train.py
    ├── metrics.json    # Latest run metrics (agent reads this)
    └── history.jsonl   # Full experiment history
```

### The Loop

```
┌─────────────────────────────────────────────────┐
│  Agent reads program.md + recent history        │
│  ↓                                              │
│  Agent modifies train.py (one targeted change)  │
│  ↓                                              │
│  python train.py → produces metrics.json        │
│  ↓                                              │
│  MAPE improved? → git commit (keep)             │
│  MAPE worse?    → git revert (discard)          │
│  ↓                                              │
│  Loop back                                      │
└─────────────────────────────────────────────────┘
```

---

## Design Decisions

**Single metric: MAPE.** Mean Absolute Percentage Error on the validation set,
averaged across all 8 grid regions. This is the only number the agent optimizes.
Region-level MAPE is logged for diagnostics but doesn't drive keep/discard.

**Fixed compute budget.** Each experiment trains XGBoost to completion (fast —
typically 10-30 seconds). No time cap needed since XGBoost isn't iterating like
neural net training. The constraint is instead on model complexity (max depth,
n_estimators) to keep experiments comparable.

**Single file.** `train.py` is ~400 lines and contains everything: feature
engineering, preprocessing, model definition, training loop, and evaluation.
The agent can modify any of it. `prepare.py` is read-only.

**Git as the ledger.** Every kept experiment is a git commit. The full history
of what worked and what didn't is in the git log.

---

## Grid Regions

| Region | Description |
|--------|-------------|
| ERCOT  | Texas grid — extreme weather sensitivity |
| PJM    | Mid-Atlantic/Midwest — largest U.S. market |
| CAISO  | California — solar/wind integration |
| MISO   | Central U.S. — wind-heavy |
| SPP    | Southern Plains — rapid growth |
| NYISO  | New York — urban demand patterns |
| ISONE  | New England — weather-driven peaks |
| SECO   | Southeast — cooling-dominated |

---

## Extending

To add a new grid region, add its EIA series ID to `prepare.py` and re-run.
The training pipeline auto-discovers regions from the data.

To change the optimization target, modify the `METRIC` constant in `train.py`.
Options: `mape`, `rmse`, `mae`, `r2`.
