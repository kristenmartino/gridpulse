#!/usr/bin/env python3
"""
journal.py — Autoresearch Journal Generator
=============================================
Reads results/history.jsonl and generates a human-readable research
journal documenting what worked, what didn't, and why.

Run manually:      python journal.py
Auto-run by agent: after every 5 experiments or at end of session

Outputs:
  results/JOURNAL.md  — The full research journal
  results/summary.json — Machine-readable summary for the dashboard
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RESULTS_DIR = Path("results")
HISTORY_FILE = RESULTS_DIR / "history.jsonl"
JOURNAL_FILE = RESULTS_DIR / "JOURNAL.md"
SUMMARY_FILE = RESULTS_DIR / "summary.json"


def load_history() -> list[dict]:
    """Load all experiments from the JSONL history."""
    if not HISTORY_FILE.exists():
        return []
    experiments = []
    for line in HISTORY_FILE.read_text().strip().split("\n"):
        if line.strip():
            experiments.append(json.loads(line))
    return experiments


def load_git_log() -> list[dict]:
    """Try to load git commit history for richer context."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--format=%H|%s|%ai"],
            capture_output=True, text=True, timeout=5
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 2)
                commits.append({
                    "hash": parts[0][:8],
                    "message": parts[1],
                    "date": parts[2].strip(),
                })
        return commits
    except Exception:
        return []


def classify_experiment(exp: dict, prev_best: float) -> dict:
    """Classify an experiment as keep/discard with context."""
    mape = exp["mape"]
    improved = mape < prev_best
    delta = prev_best - mape if improved else mape - prev_best
    delta_pct = (delta / prev_best * 100) if prev_best > 0 else 0

    return {
        "mape": mape,
        "improved": improved,
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 2),
        "status": "keep" if improved else "discard",
    }


def analyze_feature_importance_trends(experiments: list[dict]) -> dict:
    """Track how feature importance shifts across experiments."""
    if not experiments:
        return {}

    # Get feature importance from kept experiments
    kept = [e for e in experiments if e.get("feature_importance")]
    if not kept:
        return {}

    # Track top features across experiments
    feature_appearances = defaultdict(list)
    for i, exp in enumerate(kept):
        fi = exp.get("feature_importance", {})
        for feat, imp in fi.items():
            feature_appearances[feat].append({"exp": i + 1, "importance": imp})

    # Identify consistently important features
    consistent = {}
    for feat, appearances in feature_appearances.items():
        if len(appearances) >= 2:
            avg_imp = np.mean([a["importance"] for a in appearances])
            consistent[feat] = round(avg_imp, 4)

    return dict(sorted(consistent.items(), key=lambda x: -x[1])[:15])


def analyze_region_trends(experiments: list[dict]) -> dict:
    """Track per-region MAPE improvement over time."""
    if not experiments:
        return {}

    regions = {}
    first = experiments[0].get("region_mape", {})
    last = experiments[-1].get("region_mape", {})

    for region in first:
        if region in last:
            start = first[region]
            end = last[region]
            delta = start - end
            regions[region] = {
                "start": start,
                "current": end,
                "improvement": round(delta, 4),
                "improvement_pct": round((delta / start) * 100, 2) if start > 0 else 0,
            }

    return regions


def identify_patterns(experiments: list[dict]) -> list[str]:
    """Identify patterns in what worked and what didn't."""
    patterns = []

    if len(experiments) < 3:
        return ["Not enough experiments to identify patterns yet."]

    mapes = [e["mape"] for e in experiments]
    best_idx = np.argmin(mapes)
    worst_idx = np.argmax(mapes)

    # Check for plateau
    last_5 = mapes[-5:] if len(mapes) >= 5 else mapes
    if max(last_5) - min(last_5) < 0.01:
        patterns.append(
            "⚠️ **Plateau detected** — Last 5 experiments show <0.01% MAPE variation. "
            "Consider shifting strategy (e.g., from hyperparameter tuning to feature engineering, "
            "or vice versa)."
        )

    # Check for consistent improvement
    kept_count = sum(1 for e in experiments[-10:] if e["mape"] < experiments[0]["mape"])
    if kept_count >= 7:
        patterns.append(
            "🔥 **Strong improvement trend** — 7+ of last 10 experiments beat baseline. "
            "Current strategy is working well."
        )
    elif kept_count <= 2:
        patterns.append(
            "📉 **Diminishing returns** — Only 2 or fewer of last 10 experiments improved. "
            "The current search direction may be exhausted."
        )

    # Feature count trends
    feature_counts = [e.get("n_features", 0) for e in experiments if e.get("n_features")]
    if feature_counts and feature_counts[-1] < feature_counts[0] * 0.7:
        patterns.append(
            "✂️ **Feature pruning is working** — Model is performing better with fewer features, "
            "suggesting the original set had noise."
        )
    elif feature_counts and feature_counts[-1] > feature_counts[0] * 1.3:
        patterns.append(
            "📈 **Feature expansion** — More features are being used. Watch for overfitting "
            "if validation MAPE stops improving."
        )

    # Region-specific insights
    region_data = analyze_region_trends(experiments)
    hardest = max(region_data.items(), key=lambda x: x[1]["current"]) if region_data else None
    easiest = min(region_data.items(), key=lambda x: x[1]["current"]) if region_data else None
    if hardest and easiest:
        patterns.append(
            f"🗺️ **Regional gap**: {hardest[0]} is hardest ({hardest[1]['current']:.3f}% MAPE) "
            f"while {easiest[0]} is easiest ({easiest[1]['current']:.3f}%). "
            f"Targeting {hardest[0]}-specific features could yield the biggest gains."
        )

    if not patterns:
        patterns.append("No strong patterns detected yet. Keep running experiments.")

    return patterns


def generate_journal(experiments: list[dict]) -> str:
    """Generate the full research journal in Markdown."""
    if not experiments:
        return "# WattCast Research Journal\n\nNo experiments recorded yet.\n"

    baseline_mape = experiments[0]["mape"]
    best_mape = min(e["mape"] for e in experiments)
    current_mape = experiments[-1]["mape"]
    total_improvement = baseline_mape - best_mape
    improvement_pct = (total_improvement / baseline_mape) * 100 if baseline_mape > 0 else 0
    n_kept = sum(1 for i, e in enumerate(experiments) if i == 0 or e["mape"] < min(x["mape"] for x in experiments[:i]))
    total_time = sum(e.get("elapsed_seconds", 0) for e in experiments)

    git_log = load_git_log()
    region_trends = analyze_region_trends(experiments)
    feature_trends = analyze_feature_importance_trends(experiments)
    patterns = identify_patterns(experiments)

    lines = []

    # ── Header ──
    lines.append("# ⚡ WattCast Autoresearch Journal")
    lines.append(f"*Auto-generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # ── Executive Summary ──
    lines.append("## Executive Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total experiments | {len(experiments)} |")
    lines.append(f"| Improvements kept | {n_kept} |")
    lines.append(f"| Baseline MAPE | {baseline_mape:.4f}% |")
    lines.append(f"| Best MAPE | {best_mape:.4f}% |")
    lines.append(f"| Total improvement | {total_improvement:.4f}% ({improvement_pct:.1f}% relative) |")
    lines.append(f"| Compute time | {total_time:.0f}s ({total_time/60:.1f} min) |")
    lines.append("")

    # ── Key Patterns ──
    lines.append("## Key Patterns & Insights\n")
    for pattern in patterns:
        lines.append(f"- {pattern}")
    lines.append("")

    # ── Region Performance ──
    if region_trends:
        lines.append("## Region Performance Trends\n")
        lines.append("| Region | Start MAPE | Current MAPE | Improvement |")
        lines.append("|--------|-----------|-------------|-------------|")
        for region, data in sorted(region_trends.items(), key=lambda x: -x[1]["improvement_pct"]):
            emoji = "🟢" if data["improvement_pct"] > 5 else "🟡" if data["improvement_pct"] > 0 else "🔴"
            lines.append(
                f"| {emoji} {region} | {data['start']:.3f}% | {data['current']:.3f}% | "
                f"{data['improvement']:.3f}% ({data['improvement_pct']:.1f}%) |"
            )
        lines.append("")

    # ── Top Features ──
    if feature_trends:
        lines.append("## Most Consistently Important Features\n")
        lines.append("*Averaged across kept experiments:*\n")
        for i, (feat, imp) in enumerate(list(feature_trends.items())[:10], 1):
            bar = "█" * int(imp * 80)
            lines.append(f"{i:2d}. `{feat}` — {imp:.4f} {bar}")
        lines.append("")

    # ── Experiment Log ──
    lines.append("## Experiment History\n")
    
    prev_best = float("inf")
    for i, exp in enumerate(experiments):
        mape = exp["mape"]
        is_improvement = mape < prev_best
        status = "✅ KEEP" if is_improvement else "❌ DISCARD"
        delta = prev_best - mape if is_improvement else mape - prev_best
        
        lines.append(f"### Experiment {i + 1} — {status}")
        lines.append(f"- **MAPE**: {mape:.4f}%{'  ⬇️ ' + f'{delta:.4f}% improvement' if is_improvement else '  ⬆️ ' + f'{delta:.4f}% regression'}")
        
        if exp.get("region_mape"):
            worst_region = max(exp["region_mape"].items(), key=lambda x: x[1])
            best_region = min(exp["region_mape"].items(), key=lambda x: x[1])
            lines.append(f"- **Best region**: {best_region[0]} ({best_region[1]:.3f}%)")
            lines.append(f"- **Worst region**: {worst_region[0]} ({worst_region[1]:.3f}%)")
        
        if exp.get("n_features"):
            lines.append(f"- **Features used**: {exp['n_features']}")
        
        if exp.get("elapsed_seconds"):
            lines.append(f"- **Training time**: {exp['elapsed_seconds']:.1f}s")
        
        # Top 5 features for this experiment
        if exp.get("feature_importance"):
            top5 = list(exp["feature_importance"].items())[:5]
            lines.append(f"- **Top features**: {', '.join(f'`{f}`' for f, _ in top5)}")
        
        lines.append("")
        
        if is_improvement:
            prev_best = mape

    # ── What to Try Next ──
    lines.append("## Suggested Next Steps\n")
    
    if improvement_pct < 5:
        lines.append("1. **Feature engineering**: The model hasn't improved much yet. "
                     "Focus on creating new interaction features (temp × hour, humidity × season).")
        lines.append("2. **Lag structure**: Try different lag windows — the current set may "
                     "be missing important patterns.")
    elif improvement_pct < 15:
        lines.append("1. **Hyperparameter tuning**: Good feature progress. Now fine-tune "
                     "learning rate, tree depth, and regularization.")
        lines.append("2. **Feature pruning**: Remove low-importance features to reduce noise.")
    else:
        lines.append("1. **Per-region models**: Overall MAPE is strong. Train separate "
                     "models for the hardest regions.")
        lines.append("2. **Ensemble methods**: Combine XGBoost with LightGBM for "
                     "additional gains.")
    
    if region_trends:
        hardest = max(region_trends.items(), key=lambda x: x[1]["current"])
        lines.append(f"3. **Target {hardest[0]}**: This region has the highest MAPE at "
                     f"{hardest[1]['current']:.3f}%. Region-specific features could help.")
    
    lines.append("")
    lines.append("---")
    lines.append(f"*Journal generated by autoresearch journal.py — "
                f"{len(experiments)} experiments analyzed*")

    return "\n".join(lines)


def generate_summary(experiments: list[dict]) -> dict:
    """Generate a machine-readable summary for the dashboard."""
    if not experiments:
        return {"status": "no_data"}

    baseline = experiments[0]["mape"]
    best = min(e["mape"] for e in experiments)
    n_kept = sum(1 for i, e in enumerate(experiments) if i == 0 or e["mape"] < min(x["mape"] for x in experiments[:i]))

    return {
        "total_experiments": len(experiments),
        "kept": n_kept,
        "discarded": len(experiments) - n_kept,
        "baseline_mape": baseline,
        "best_mape": best,
        "improvement_pct": round(((baseline - best) / baseline) * 100, 2) if baseline > 0 else 0,
        "region_trends": analyze_region_trends(experiments),
        "top_features": analyze_feature_importance_trends(experiments),
        "patterns": identify_patterns(experiments),
        "generated_at": datetime.now().isoformat(),
    }


def main():
    print("📓 Generating research journal...")

    experiments = load_history()
    if not experiments:
        print("  No experiments found in results/history.jsonl")
        print("  Run some experiments first: python train.py")
        return

    # Generate journal
    journal = generate_journal(experiments)
    JOURNAL_FILE.write_text(journal)
    print(f"  ✅ Journal saved to {JOURNAL_FILE}")
    print(f"     {len(experiments)} experiments documented")

    # Generate summary
    summary = generate_summary(experiments)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"  ✅ Summary saved to {SUMMARY_FILE}")

    # Print highlights
    print(f"\n  Baseline MAPE:  {summary['baseline_mape']:.4f}%")
    print(f"  Best MAPE:      {summary['best_mape']:.4f}%")
    print(f"  Improvement:    {summary['improvement_pct']:.1f}%")
    print(f"  Kept:           {summary['kept']} / {summary['total_experiments']}")
    print()


if __name__ == "__main__":
    main()
