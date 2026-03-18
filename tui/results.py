"""Results management for the experiment orchestrator.

Handles reading, writing, and formatting results.tsv for the experiment loop.
"""

import csv
import os
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """A single experiment result."""
    exp: str           # "exp0", "exp1", etc.
    description: str   # short description
    val_bpb: float     # validation bits per byte (0.0 for crashes)
    peak_mem_gb: float # peak memory in GB (0.0 for crashes)
    tok_sec: int       # tokens per second
    mfu: float         # model FLOPS utilization %
    steps: int         # training steps completed
    status: str        # "baseline", "keep", "discard", "crash"
    notes: str         # extra notes


# TSV header matching the format used by characterization sessions
HEADER = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\n"


def init_results_tsv(path: str = "results.tsv") -> None:
    """Create results.tsv with header if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(HEADER)


def append_result(path: str, result: ExperimentResult) -> None:
    """Append an experiment result to results.tsv."""
    if not os.path.exists(path):
        init_results_tsv(path)

    with open(path, "a") as f:
        f.write(
            f"{result.exp}\t"
            f"{result.description}\t"
            f"{result.val_bpb:.6f}\t"
            f"{result.peak_mem_gb:.1f}\t"
            f"{result.tok_sec}\t"
            f"{result.mfu:.1f}\t"
            f"{result.steps}\t"
            f"{result.status}\t"
            f"{result.notes}\n"
        )


def load_results(path: str = "results.tsv") -> list[ExperimentResult]:
    """Load all experiment results from results.tsv."""
    if not os.path.exists(path):
        return []

    results = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                results.append(ExperimentResult(
                    exp=row.get("exp", ""),
                    description=row.get("description", ""),
                    val_bpb=float(row.get("val_bpb", 0)),
                    peak_mem_gb=float(row.get("peak_mem_gb", 0)),
                    tok_sec=int(row.get("tok_sec", 0)),
                    mfu=float(row.get("mfu", 0)),
                    steps=int(row.get("steps", 0)),
                    status=row.get("status", ""),
                    notes=row.get("notes", ""),
                ))
            except (ValueError, TypeError):
                continue
    return results


def get_best_result(path: str = "results.tsv") -> tuple[float, str]:
    """Get the best val_bpb and experiment name from results.

    Only considers "keep" and "baseline" experiments.
    Returns (best_val_bpb, best_exp_name). Returns (inf, "none") if no results.
    """
    results = load_results(path)
    best_bpb = float("inf")
    best_exp = "none"

    for r in results:
        if r.status in ("keep", "baseline") and r.val_bpb > 0:
            if r.val_bpb < best_bpb:
                best_bpb = r.val_bpb
                best_exp = f"{r.exp} ({r.description})"

    return best_bpb, best_exp


def next_experiment_number(path: str = "results.tsv") -> int:
    """Get the next experiment number based on existing results."""
    results = load_results(path)
    if not results:
        return 0

    max_num = -1
    for r in results:
        try:
            num = int(r.exp.replace("exp", ""))
            max_num = max(max_num, num)
        except ValueError:
            continue
    return max_num + 1


def format_history_for_prompt(path: str = "results.tsv") -> str:
    """Format results as a readable table for the LLM prompt."""
    results = load_results(path)
    if not results:
        return "No experiments yet."

    lines = []
    lines.append(f"{'Exp':<6} {'Status':<9} {'val_bpb':>8} {'Mem(GB)':>8} {'Tok/s':>8} {'MFU':>6} {'Steps':>6}  Description")
    lines.append("-" * 90)

    for r in results:
        bpb_str = f"{r.val_bpb:.4f}" if r.val_bpb > 0 else "crash"
        lines.append(
            f"{r.exp:<6} {r.status:<9} {bpb_str:>8} {r.peak_mem_gb:>7.1f} "
            f"{r.tok_sec:>8} {r.mfu:>5.1f}% {r.steps:>6}  {r.description}"
        )

    return "\n".join(lines)
