"""Load experiment results from results.tsv."""

import csv
import os
from dataclasses import dataclass


@dataclass
class Experiment:
    exp: str = ""
    description: str = ""
    val_bpb: str = ""
    peak_mem_gb: str = ""
    tok_sec: str = ""
    mfu: str = ""
    steps: str = ""
    status: str = ""
    notes: str = ""


def load_experiments(tsv_path: str = "results.tsv") -> list[Experiment]:
    """Load experiments from results.tsv if it exists."""
    if not os.path.exists(tsv_path):
        return []

    experiments = []
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            experiments.append(Experiment(
                exp=row.get('exp', ''),
                description=row.get('description', ''),
                val_bpb=row.get('val_bpb', ''),
                peak_mem_gb=row.get('peak_mem_gb', ''),
                tok_sec=row.get('tok_sec', ''),
                mfu=row.get('mfu', ''),
                steps=row.get('steps', ''),
                status=row.get('status', ''),
                notes=row.get('notes', ''),
            ))
    return experiments
