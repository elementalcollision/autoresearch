#!/usr/bin/env python3
"""
Cross-dataset comparison and analysis tool.

Reads results from results/<dataset>/results.tsv for each completed dataset
and generates comparative visualizations and analysis.

Usage:
    uv run compare_datasets.py                    # Full comparison
    uv run compare_datasets.py --output chart.png  # Custom output path
    uv run compare_datasets.py --summary           # Text summary only
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"

@dataclass
class Experiment:
    name: str
    description: str
    val_bpb: float
    peak_mem_gb: float
    tok_sec: float
    mfu: float
    steps: int
    status: str
    notes: str = ""


@dataclass
class DatasetRun:
    dataset: str
    experiments: list = field(default_factory=list)

    @property
    def baseline(self):
        for e in self.experiments:
            if e.status == "baseline":
                return e
        return self.experiments[0] if self.experiments else None

    @property
    def best(self):
        keeps = [e for e in self.experiments if e.status in ("keep", "baseline") and e.val_bpb > 0]
        return min(keeps, key=lambda e: e.val_bpb) if keeps else None

    @property
    def kept(self):
        return [e for e in self.experiments if e.status == "keep"]

    @property
    def discarded(self):
        return [e for e in self.experiments if e.status == "discard"]

    @property
    def crashed(self):
        return [e for e in self.experiments if e.status == "crash"]

    @property
    def improvement_pct(self):
        b = self.baseline
        best = self.best
        if b and best and b.val_bpb > 0:
            return (b.val_bpb - best.val_bpb) / b.val_bpb * 100
        return 0.0

    @property
    def optimal_config(self):
        """Extract the optimal hyperparameter changes from the keep chain."""
        changes = {}
        for e in self.kept:
            desc = e.description.lower()
            # Parse "Increase/Decrease X from Y to Z" patterns
            for param in ["aspect_ratio", "matrix_lr", "scalar_lr", "embedding_lr",
                          "unembedding_lr", "weight_decay", "warmdown_ratio",
                          "warmup_ratio", "device_batch_size", "total_batch_size",
                          "head_dim", "depth"]:
                if param.lower().replace("_", "") in desc.replace("_", "").lower():
                    # Extract the "to X" value
                    parts = desc.split(" to ")
                    if len(parts) >= 2:
                        try:
                            val = parts[-1].strip().rstrip(".")
                            changes[param.upper()] = val
                        except (ValueError, IndexError):
                            pass
        return changes


def load_results(dataset_name):
    """Load results.tsv for a dataset."""
    tsv_path = RESULTS_DIR / dataset_name / "results.tsv"
    if not tsv_path.exists():
        return None

    run = DatasetRun(dataset=dataset_name)
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("exp\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            try:
                exp = Experiment(
                    name=parts[0],
                    description=parts[1],
                    val_bpb=float(parts[2]),
                    peak_mem_gb=float(parts[3]),
                    tok_sec=float(parts[4]),
                    mfu=float(parts[5]),
                    steps=int(parts[6]),
                    status=parts[7],
                    notes=parts[8] if len(parts) > 8 else "",
                )
                run.experiments.append(exp)
            except (ValueError, IndexError):
                continue

    return run if run.experiments else None


def load_all_results():
    """Load results for all datasets that have them."""
    runs = {}
    if not RESULTS_DIR.exists():
        return runs

    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir():
            run = load_results(d.name)
            if run:
                runs[d.name] = run
    return runs


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(runs):
    """Print a text comparison summary."""
    if not runs:
        print("No results found. Run experiments first.")
        return

    print("\n" + "=" * 80)
    print("  CROSS-DATASET COMPARISON")
    print("=" * 80)

    # Overview table
    print(f"\n  {'Dataset':<20} {'Exps':>5} {'Kept':>5} {'Crash':>5} {'Baseline':>10} {'Best':>10} {'Improv':>8} {'Mem GB':>8}")
    print("  " + "-" * 73)

    for name, run in runs.items():
        bl = run.baseline
        best = run.best
        print(f"  {name:<20} {len(run.experiments):>5} {len(run.kept):>5} {len(run.crashed):>5} "
              f"{bl.val_bpb:>10.6f} {best.val_bpb:>10.6f} {run.improvement_pct:>7.1f}% {best.peak_mem_gb:>7.1f}")

    # Optimal configs comparison
    print(f"\n  Optimal Hyperparameter Comparison")
    print("  " + "-" * 73)

    all_params = set()
    for run in runs.values():
        all_params.update(run.optimal_config.keys())

    if all_params:
        print(f"  {'Parameter':<25}", end="")
        for name in runs:
            print(f" {name:<15}", end="")
        print()

        for param in sorted(all_params):
            print(f"  {param:<25}", end="")
            for name, run in runs.items():
                val = run.optimal_config.get(param, "default")
                print(f" {val:<15}", end="")
            print()

    # Key insights
    print(f"\n  Key Observations")
    print("  " + "-" * 73)

    if len(runs) >= 2:
        # Find which dataset had the biggest improvement
        best_improv = max(runs.items(), key=lambda x: x[1].improvement_pct)
        print(f"  - Biggest improvement: {best_improv[0]} ({best_improv[1].improvement_pct:.1f}%)")

        # Find which dataset had the best absolute val_bpb
        best_abs = min(runs.items(), key=lambda x: x[1].best.val_bpb)
        print(f"  - Best absolute val_bpb: {best_abs[0]} ({best_abs[1].best.val_bpb:.6f})")

        # Find which dataset had the highest keep rate
        best_keep = max(runs.items(), key=lambda x: len(x[1].kept) / max(len(x[1].experiments), 1))
        keep_rate = len(best_keep[1].kept) / len(best_keep[1].experiments) * 100
        print(f"  - Highest keep rate: {best_keep[0]} ({keep_rate:.0f}%)")

        # Check if optimal configs differ
        configs = {name: run.optimal_config for name, run in runs.items()}
        shared_params = set.intersection(*[set(c.keys()) for c in configs.values()]) if configs else set()
        differing = [p for p in shared_params if len(set(configs[n].get(p, "default") for n in runs)) > 1]

        if differing:
            print(f"  - Parameters that differ across datasets: {', '.join(sorted(differing))}")
        else:
            print(f"  - All tuned parameters agree across datasets (hyperparameters transfer!)")

    print()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_chart(runs, output_path):
    """Generate a multi-panel comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed. Run: uv pip install matplotlib")
        return

    n_datasets = len(runs)
    if n_datasets == 0:
        print("No results to chart.")
        return

    colors = plt.cm.tab10.colors[:n_datasets]
    dataset_colors = dict(zip(runs.keys(), colors))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Multi-Dataset Experiment Comparison", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: val_bpb convergence curves (relative to baseline)
    ax1 = axes[0, 0]
    for name, run in runs.items():
        bl = run.baseline.val_bpb if run.baseline else 1.0
        keeps = [e for e in run.experiments if e.status in ("keep", "baseline") and e.val_bpb > 0]
        if keeps:
            # Build cumulative best curve
            best_so_far = bl
            xs, ys = [0], [0.0]
            for i, e in enumerate(keeps):
                if e.val_bpb < best_so_far:
                    best_so_far = e.val_bpb
                xs.append(i + 1)
                ys.append((bl - best_so_far) / bl * 100)
            ax1.plot(xs, ys, "o-", color=dataset_colors[name], label=name, markersize=4)

    ax1.set_xlabel("Kept experiments")
    ax1.set_ylabel("Improvement over baseline (%)")
    ax1.set_title("Convergence Curves (Relative Improvement)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Best val_bpb bar chart
    ax2 = axes[0, 1]
    names = list(runs.keys())
    baselines = [runs[n].baseline.val_bpb for n in names]
    bests = [runs[n].best.val_bpb for n in names]
    x = range(len(names))

    bars_bl = ax2.bar([i - 0.15 for i in x], baselines, 0.3, label="Baseline", color="lightgray", edgecolor="gray")
    bars_best = ax2.bar([i + 0.15 for i in x], bests, 0.3, label="Best",
                        color=[dataset_colors[n] for n in names], edgecolor="black")

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("val_bpb")
    ax2.set_title("Baseline vs Best val_bpb")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Add improvement labels
    for i, (bl, best) in enumerate(zip(baselines, bests)):
        improv = (bl - best) / bl * 100
        ax2.annotate(f"-{improv:.1f}%", xy=(i + 0.15, best), xytext=(0, -15),
                     textcoords="offset points", ha="center", fontsize=8, fontweight="bold",
                     color="green")

    # Panel 3: Memory vs Performance scatter
    ax3 = axes[1, 0]
    for name, run in runs.items():
        keeps = [e for e in run.experiments if e.status == "keep" and e.val_bpb > 0]
        if keeps:
            mems = [e.peak_mem_gb for e in keeps]
            vals = [e.val_bpb for e in keeps]
            ax3.scatter(mems, vals, c=[dataset_colors[name]], label=name, alpha=0.7, s=40)
            # Highlight best
            best = run.best
            if best:
                ax3.scatter([best.peak_mem_gb], [best.val_bpb], c=[dataset_colors[name]],
                            marker="*", s=200, edgecolors="black", linewidth=1, zorder=5)

    ax3.set_xlabel("Peak Memory (GB)")
    ax3.set_ylabel("val_bpb")
    ax3.set_title("Pareto Frontier: Memory vs Performance")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")

    headers = ["Dataset", "Exps", "Kept", "Crash", "Keep %", "Best BPB", "Improv %", "Steps"]
    table_data = []
    for name, run in runs.items():
        keep_rate = len(run.kept) / max(len(run.experiments), 1) * 100
        table_data.append([
            name,
            str(len(run.experiments)),
            str(len(run.kept)),
            str(len(run.crashed)),
            f"{keep_rate:.0f}%",
            f"{run.best.val_bpb:.4f}",
            f"{run.improvement_pct:.1f}%",
            str(run.best.steps),
        ])

    table = ax4.table(cellText=table_data, colLabels=headers, loc="center",
                      cellLoc="center", colColours=["#f0f0f0"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax4.set_title("Summary Statistics", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset comparison tool")
    parser.add_argument("--output", type=str, default="dataset_comparison.png",
                        help="Output chart path (default: dataset_comparison.png)")
    parser.add_argument("--summary", action="store_true",
                        help="Print text summary only (no chart)")

    args = parser.parse_args()

    runs = load_all_results()

    if not runs:
        print("No results found in results/*/results.tsv")
        print("Run experiments first, or copy results.tsv files to results/<dataset>/")
        sys.exit(1)

    print_summary(runs)

    if not args.summary:
        generate_chart(runs, args.output)


if __name__ == "__main__":
    main()
