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
import hashlib
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
# Content fingerprinting
# ---------------------------------------------------------------------------

def _fingerprint_data_dir(data_dir):
    """Generate a content fingerprint from the first shard's first document.

    Returns a dict with hash + sample text for human verification.
    """
    shard_path = data_dir / "shard_00000.parquet"
    if not shard_path.exists():
        return None

    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(shard_path))
        rg = pf.read_row_group(0)
        first_doc = rg.column("text").to_pylist()[0]
        doc_hash = hashlib.sha256(first_doc.encode("utf-8")).hexdigest()[:16]
        sample = first_doc[:200].replace("\n", " ")
        return {"hash": doc_hash, "sample": sample}
    except Exception as e:
        print(f"  WARNING: Could not fingerprint data: {e}")
        return None


def _validate_fingerprint(profile_dir, expected_name):
    """Check if a profile's fingerprint matches its stored identity.

    Returns True if valid, False if mismatched or missing fingerprint.
    """
    meta_path = profile_dir / "meta.json"
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    stored_fp = meta.get("fingerprint")
    if not stored_fp:
        # No fingerprint = old profile, can't trust it
        print(f"  WARNING: Profile '{expected_name}' has no fingerprint — cannot validate")
        return False

    # Re-fingerprint the profile's data
    data_dir = profile_dir / "data"
    current_fp = _fingerprint_data_dir(data_dir)
    if not current_fp:
        return False

    if current_fp["hash"] != stored_fp["hash"]:
        print(f"  ERROR: Profile '{expected_name}' fingerprint mismatch!")
        print(f"    Stored:  {stored_fp['sample'][:80]}...")
        print(f"    Actual:  {current_fp['sample'][:80]}...")
        return False

    return True


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

def save_profile(name, force=False):
    """Save current data + tokenizer as a named profile with fingerprint."""
    profile_dir = PROFILES_DIR / name
    if profile_dir.exists() and not force:
        print(f"  Profile '{name}' already exists, skipping save (use --rebuild-profiles to force)")
        return

    # If forcing, remove the old profile
    if profile_dir.exists() and force:
        print(f"  Removing stale profile '{name}'...")
        shutil.rmtree(profile_dir)

    profile_dir.mkdir(parents=True, exist_ok=True)

    if DATA_DIR.exists():
        data_dest = profile_dir / "data"
        print(f"  Saving data → {data_dest}")
        shutil.copytree(DATA_DIR, data_dest)

    if TOKENIZER_DIR.exists():
        tok_dest = profile_dir / "tokenizer"
        print(f"  Saving tokenizer → {tok_dest}")
        shutil.copytree(TOKENIZER_DIR, tok_dest)

    # Fingerprint the data for validation
    fingerprint = _fingerprint_data_dir(profile_dir / "data")

    # Metadata with fingerprint
    meta = {
        "dataset": name,
        "created": datetime.now().isoformat(),
        "shards": len(list((profile_dir / "data").glob("*.parquet"))) if (profile_dir / "data").exists() else 0,
        "fingerprint": fingerprint,
    }
    with open(profile_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if fingerprint:
        print(f"  Profile '{name}' saved (fingerprint: {fingerprint['hash']})")
        print(f"    Sample: {fingerprint['sample'][:80]}...")
    else:
        print(f"  Profile '{name}' saved (no fingerprint — verify manually)")


def load_profile(name):
    """Restore a named profile as the active data + tokenizer.

    Validates the fingerprint before loading. Returns False if the
    profile is missing or its content doesn't match its claimed identity.
    """
    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        return False

    # Validate fingerprint before trusting the profile
    if not _validate_fingerprint(profile_dir, name):
        print(f"  Profile '{name}' failed validation — will not load")
        print(f"  Run with --rebuild-profiles to fix")
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

    # Post-load verification: fingerprint the loaded data against the profile
    loaded_fp = _fingerprint_data_dir(DATA_DIR)
    with open(profile_dir / "meta.json") as f:
        meta = json.load(f)
    stored_fp = meta.get("fingerprint", {})

    if loaded_fp and stored_fp and loaded_fp["hash"] == stored_fp["hash"]:
        print(f"  Loaded profile '{name}' ✓ (verified: {loaded_fp['hash']})")
    else:
        print(f"  Loaded profile '{name}' (fingerprint verification skipped)")

    return True


def profile_exists(name):
    """Check if a valid profile exists (has data dir and passes fingerprint)."""
    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        return False
    if not (profile_dir / "data").exists():
        return False
    # Check for fingerprint — profiles without one are considered invalid
    meta_path = profile_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("fingerprint"):
            return False  # No fingerprint = untrusted legacy profile
    return True


def delete_profile(name):
    """Delete a profile."""
    profile_dir = PROFILES_DIR / name
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        print(f"  Deleted profile '{name}'")


def list_profiles():
    """List all saved profiles with metadata and validation status."""
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
            fp = meta.get("fingerprint", {})
            valid = _validate_fingerprint(d, d.name) if fp else False
            profiles.append({
                "name": d.name,
                "shards": meta.get("shards", "?"),
                "created": meta.get("created", "unknown"),
                "fingerprint": fp.get("hash", "none"),
                "sample": fp.get("sample", "")[:60],
                "valid": valid,
            })
    return profiles


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_climbmix(num_shards=10):
    """Prepare the default climbmix dataset.

    Always validates the profile before trusting it. If the profile
    exists but fails validation, it's rebuilt from scratch.
    """
    # Check if we already have a valid profile
    if profile_exists("climbmix"):
        print("  climbmix profile exists, loading...")
        if load_profile("climbmix"):
            return True
        else:
            print("  climbmix profile failed validation, rebuilding...")
            delete_profile("climbmix")

    # Download fresh — the only guaranteed way to get real climbmix
    # (backup restore was removed: the backup_fineweb-edu dir contained
    # FineWeb-Edu data, not climbmix, causing profile contamination)
    print("  Downloading climbmix shards (fresh)...")
    # Clear any existing data first to avoid contamination
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if TOKENIZER_DIR.exists():
        shutil.rmtree(TOKENIZER_DIR)

    result = subprocess.run(
        ["uv", "run", "prepare.py", f"--num-shards={num_shards}"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode == 0:
        fp = _fingerprint_data_dir(DATA_DIR)
        if fp:
            print(f"  Downloaded data sample: {fp['sample'][:80]}...")
        save_profile("climbmix", force=True)
        return True
    else:
        print(f"  Download failed")
        return False


def prepare_alternative(dataset_name, num_shards=10, num_source=3):
    """Prepare an alternative dataset via convert_dataset.py."""
    if profile_exists(dataset_name):
        print(f"  {dataset_name} profile exists, loading...")
        if load_profile(dataset_name):
            return True
        else:
            print(f"  {dataset_name} profile failed validation, rebuilding...")
            delete_profile(dataset_name)

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

    fp = _fingerprint_data_dir(DATA_DIR)
    if fp:
        print(f"  Converted data sample: {fp['sample'][:80]}...")
    save_profile(dataset_name, force=True)
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
    """Run the autonomous agent for a dataset.

    Uses the headless orchestrator (no TUI) so the run survives terminal
    disconnects, SSH timeouts, and overnight unattended operation.
    The TUI dashboard (dashboard.py --agent) is for interactive use only.
    """
    from tui.headless import run_headless

    results_dir = get_results_dir(dataset_name)
    results_tsv = str(results_dir / "results.tsv")
    run_tag = f"{tag}-{dataset_name}"

    print(f"\n{'='*60}")
    print(f"  Running agent (headless): {dataset_name}")
    print(f"  Tag: {run_tag}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Results: {results_tsv}")
    print(f"{'='*60}\n")

    try:
        return run_headless(
            training_script="train_mlx.py",
            results_path=results_tsv,
            tag=run_tag,
            max_experiments=max_experiments,
        )
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

    # Show profiles with validation
    profiles = list_profiles()
    if profiles:
        print(f"\n  Saved profiles ({PROFILES_DIR}):")
        for p in profiles:
            status = "✓" if p["valid"] else "✗"
            fp = p.get("fingerprint", "none")
            print(f"    {status} {p['name']}: {p['shards']} shards, fp={fp}, created {p['created'][:10]}")
            if p.get("sample"):
                print(f"      Sample: {p['sample']}...")
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
    parser.add_argument("--rebuild-profiles", action="store_true",
                        help="Delete and rebuild all profiles from scratch")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all profiles and report any mismatches")

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

    # --- Validate profiles ---
    if args.validate:
        print("\n  Profile Validation")
        print("  " + "=" * 70)
        profiles = list_profiles()
        if not profiles:
            print("  No profiles found.")
        else:
            for p in profiles:
                status = "✓ VALID" if p["valid"] else "✗ INVALID"
                fp = p["fingerprint"]
                print(f"  {p['name']:<20} {status:<12} fp={fp:<18} {p['sample']}")
        print("  " + "=" * 70)
        return

    # --- Rebuild profiles ---
    if args.rebuild_profiles:
        print("\n  Rebuilding all profiles...")
        if PROFILES_DIR.exists():
            for d in PROFILES_DIR.iterdir():
                if d.is_dir():
                    print(f"  Deleting profile '{d.name}'...")
                    shutil.rmtree(d)
        print("  All profiles deleted. They will be rebuilt on next suite run.")
        print("  Run: uv run run_suite.py --prepare-only")
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

        # Run agent (headless — no TUI, survives terminal disconnect)
        run_agent(dataset_name, tag, args.max_experiments)

    # --- Final status ---
    print_status()


if __name__ == "__main__":
    main()
