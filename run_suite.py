#!/usr/bin/env python3
"""
Multi-dataset experiment suite runner.

Orchestrates the full autoresearch experiment loop across multiple datasets,
keeping each dataset's results cleanly isolated.

Usage:
    # Run full suite (convert → tokenize → agent run for each dataset)
    uv run run_suite.py

    # Run a single dataset
    uv run run_suite.py --dataset fineweb-edu

    # List available datasets and their status
    uv run run_suite.py --status

    # Skip datasets that already have results
    uv run run_suite.py --skip-completed

    # Customize per-dataset experiment count
    uv run run_suite.py --max-experiments 80

Each dataset gets:
  - Its own data + tokenizer profile in ~/.cache/autoresearch/profile_<name>/
  - Its own results file in results/<name>/results.tsv
  - Its own git branch: autoresearch/<tag>-<name>
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
RESULTS_DIR = PROJECT_ROOT / "results"
PROFILES_DIR = CACHE_DIR / "profiles"

# Dataset run order (priority order from plan)
DATASET_ORDER = [
    "climbmix",
    "fineweb-edu",
    "cosmopedia-v2",
    "slimpajama",
    "fineweb-edu-high",
    "python-edu",
]

# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

def save_profile(name):
    """Save current data + tokenizer as a named profile."""
    profile_dir = PROFILES_DIR / name
    if profile_dir.exists():
        print(f"  Profile '{name}' already exists, skipping save")
        return

    profile_dir.mkdir(parents=True, exist_ok=True)

    if DATA_DIR.exists():
        data_dest = profile_dir / "data"
        print(f"  Saving data → {data_dest}")
        shutil.copytree(DATA_DIR, data_dest)

    if TOKENIZER_DIR.exists():
        tok_dest = profile_dir / "tokenizer"
        print(f"  Saving tokenizer → {tok_dest}")
        shutil.copytree(TOKENIZER_DIR, tok_dest)

    # Metadata
    meta = {
        "dataset": name,
        "created": datetime.now().isoformat(),
        "shards": len(list((profile_dir / "data").glob("*.parquet"))) if (profile_dir / "data").exists() else 0,
    }
    with open(profile_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Profile '{name}' saved")


def load_profile(name):
    """Restore a named profile as the active data + tokenizer."""
    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        return False

    # Clear current
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if TOKENIZER_DIR.exists():
        shutil.rmtree(TOKENIZER_DIR)

    # Restore from profile
    data_src = profile_dir / "data"
    if data_src.exists():
        shutil.copytree(data_src, DATA_DIR)

    tok_src = profile_dir / "tokenizer"
    if tok_src.exists():
        shutil.copytree(tok_src, TOKENIZER_DIR)

    print(f"  Loaded profile '{name}'")
    return True


def profile_exists(name):
    """Check if a profile has been saved."""
    return (PROFILES_DIR / name).exists()


def list_profiles():
    """List all saved profiles with metadata."""
    if not PROFILES_DIR.exists():
        return []

    profiles = []
    for d in sorted(PROFILES_DIR.iterdir()):
        if d.is_dir():
            meta_path = d / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            profiles.append({
                "name": d.name,
                "shards": meta.get("shards", "?"),
                "created": meta.get("created", "unknown"),
            })
    return profiles


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_climbmix(num_shards=10):
    """Prepare the default climbmix dataset."""
    # Check if we already have a profile
    if profile_exists("climbmix"):
        print("  climbmix profile exists, loading...")
        load_profile("climbmix")
        return True

    # Check if current data is climbmix (from backup)
    backup_dir = CACHE_DIR / "backup_fineweb-edu"
    if backup_dir.exists():
        print("  Restoring climbmix from backup...")
        result = subprocess.run(
            ["uv", "run", "convert_dataset.py", "--restore"],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            save_profile("climbmix")
            return True
        else:
            print(f"  Restore failed: {result.stderr}")

    # Download fresh
    print("  Downloading climbmix shards...")
    result = subprocess.run(
        ["uv", "run", "prepare.py", f"--num-shards={num_shards}"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        save_profile("climbmix")
        return True
    else:
        print(f"  Download failed: {result.stderr}")
        return False


def prepare_alternative(dataset_name, num_shards=10, num_source=3):
    """Prepare an alternative dataset via convert_dataset.py."""
    if profile_exists(dataset_name):
        print(f"  {dataset_name} profile exists, loading...")
        load_profile(dataset_name)
        return True

    print(f"  Converting {dataset_name}...")
    result = subprocess.run(
        [
            "uv", "run", "convert_dataset.py", dataset_name,
            f"--num-shards={num_shards}",
            f"--num-source={num_source}",
            "--skip-backup",
        ],
        cwd=PROJECT_ROOT,
        timeout=3600,  # 1 hour max for download
    )
    if result.returncode != 0:
        print(f"  Conversion failed for {dataset_name}")
        return False

    # Train tokenizer
    print(f"  Training tokenizer for {dataset_name}...")
    result = subprocess.run(
        ["uv", "run", "prepare.py", "--num-shards=0"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"  Tokenizer training failed for {dataset_name}")
        return False

    save_profile(dataset_name)
    return True


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def get_results_dir(dataset_name):
    """Get the results directory for a dataset."""
    d = RESULTS_DIR / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_results(dataset_name):
    """Check if a dataset already has experiment results."""
    tsv = RESULTS_DIR / dataset_name / "results.tsv"
    if not tsv.exists():
        return False
    # Count non-header lines
    with open(tsv) as f:
        lines = [l for l in f if l.strip() and not l.startswith("exp\t")]
    return len(lines) > 0


def count_experiments(dataset_name):
    """Count completed experiments for a dataset."""
    tsv = RESULTS_DIR / dataset_name / "results.tsv"
    if not tsv.exists():
        return 0
    with open(tsv) as f:
        return sum(1 for l in f if l.strip() and not l.startswith("exp\t"))


def run_agent(dataset_name, tag, max_experiments=80):
    """Run the autonomous agent for a dataset."""
    results_dir = get_results_dir(dataset_name)

    print(f"\n{'='*60}")
    print(f"  Running agent: {dataset_name}")
    print(f"  Tag: {tag}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}\n")

    # Run the dashboard in agent mode
    cmd = [
        "uv", "run", "--extra", "mlx", "--extra", "agent",
        "dashboard.py",
        "--agent",
        f"--tag={tag}-{dataset_name}",
        f"--max={max_experiments}",
    ]

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=max_experiments * 400)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Agent timed out after {max_experiments * 400}s")
        return False
    except KeyboardInterrupt:
        print(f"\n  Agent interrupted by user")
        return False


# ---------------------------------------------------------------------------
# Status and reporting
# ---------------------------------------------------------------------------

def print_status():
    """Print status of all datasets."""
    print("\n  Multi-Dataset Experiment Status")
    print("  " + "=" * 58)
    print(f"  {'Dataset':<20} {'Profile':<10} {'Experiments':<12} {'Best val_bpb':<14}")
    print("  " + "-" * 58)

    for name in DATASET_ORDER:
        has_profile = "yes" if profile_exists(name) else "no"
        n_exp = count_experiments(name)

        best = "—"
        tsv = RESULTS_DIR / name / "results.tsv"
        if tsv.exists():
            with open(tsv) as f:
                vals = []
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 8 and parts[7] in ("keep", "baseline"):
                        try:
                            vals.append(float(parts[2]))
                        except ValueError:
                            pass
                if vals:
                    best = f"{min(vals):.6f}"

        print(f"  {name:<20} {has_profile:<10} {n_exp:<12} {best:<14}")

    print("  " + "=" * 58)

    # Show profiles
    profiles = list_profiles()
    if profiles:
        print(f"\n  Saved profiles ({PROFILES_DIR}):")
        for p in profiles:
            print(f"    {p['name']}: {p['shards']} shards, created {p['created'][:10]}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset experiment suite runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=DATASET_ORDER,
                        help="Run a single dataset (default: run all in order)")
    parser.add_argument("--status", action="store_true",
                        help="Show status of all datasets")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip datasets that already have results")
    parser.add_argument("--max-experiments", type=int, default=80,
                        help="Max experiments per dataset (default: 80)")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Training shards per dataset (default: 10)")
    parser.add_argument("--num-source", type=int, default=3,
                        help="Source files to download per dataset (default: 3)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag (default: today's date)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare datasets, don't run experiments")
    parser.add_argument("--save-profile", type=str, metavar="NAME",
                        help="Save current data+tokenizer as a named profile")
    parser.add_argument("--load-profile", type=str, metavar="NAME",
                        help="Load a named profile as active data+tokenizer")

    args = parser.parse_args()

    # --- Profile management ---
    if args.save_profile:
        print(f"Saving profile '{args.save_profile}'...")
        save_profile(args.save_profile)
        return

    if args.load_profile:
        print(f"Loading profile '{args.load_profile}'...")
        if load_profile(args.load_profile):
            print("Done! Ready to train.")
        else:
            print(f"ERROR: Profile '{args.load_profile}' not found.")
            sys.exit(1)
        return

    # --- Status ---
    if args.status:
        print_status()
        return

    # --- Determine run tag ---
    tag = args.tag or datetime.now().strftime("%b%d").lower()

    # --- Determine which datasets to run ---
    datasets = [args.dataset] if args.dataset else DATASET_ORDER

    print(f"\nMulti-Dataset Experiment Suite")
    print(f"  Tag: {tag}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Max experiments per dataset: {args.max_experiments}")
    print(f"  Shards per dataset: {args.num_shards}")
    print()

    # --- Run each dataset ---
    for i, dataset_name in enumerate(datasets):
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(datasets)}] Dataset: {dataset_name}")
        print(f"{'#'*60}")

        # Skip if completed
        if args.skip_completed and has_results(dataset_name):
            n = count_experiments(dataset_name)
            print(f"  Skipping — already has {n} experiments")
            continue

        # Prepare data
        print(f"\n  Preparing {dataset_name}...")
        if dataset_name == "climbmix":
            success = prepare_climbmix(args.num_shards)
        else:
            # Source file counts tuned per dataset
            num_source = args.num_source
            if dataset_name == "slimpajama":
                num_source = max(6, args.num_source)  # smaller files, need more
            elif dataset_name == "python-edu":
                num_source = max(5, args.num_source)  # smaller files
            success = prepare_alternative(dataset_name, args.num_shards, num_source)

        if not success:
            print(f"  FAILED to prepare {dataset_name}, skipping")
            continue

        if args.prepare_only:
            print(f"  Prepared {dataset_name} (--prepare-only, skipping agent run)")
            continue

        # Run agent
        run_agent(dataset_name, tag, args.max_experiments)

        # Copy results to dataset-specific directory
        # The agent writes results.tsv in the project root via the orchestrator
        src_results = PROJECT_ROOT / "results.tsv"
        if src_results.exists():
            dest = get_results_dir(dataset_name) / "results.tsv"
            shutil.copy2(src_results, dest)
            print(f"  Results saved to {dest}")

    # --- Final status ---
    print_status()


if __name__ == "__main__":
    main()
