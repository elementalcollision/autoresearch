"""
Dataset conversion script for autoresearch experiments.

Downloads alternative datasets from HuggingFace and converts them to the
shard format expected by prepare.py (parquet files with a 'text' column,
named shard_NNNNN.parquet).

Usage:
    # FineWeb-Edu (default: 10 training shards + 1 val shard)
    uv run convert_dataset.py fineweb-edu

    # More shards for longer runs
    uv run convert_dataset.py fineweb-edu --num-shards 50

    # Dry run (show what would be downloaded)
    uv run convert_dataset.py fineweb-edu --dry-run

    # Restore original climbmix dataset
    uv run convert_dataset.py --restore

The script:
  1. Backs up current data + tokenizer to ~/.cache/autoresearch/backup_<name>/
  2. Downloads source parquet files from HuggingFace
  3. Re-chunks into ~88MB shards matching climbmix format
  4. Pins the last shard as shard_06542 (validation)
  5. Removes the old tokenizer so prepare.py retrains it

After conversion, run: uv run prepare.py
Then run experiments as normal.
"""

import os
import sys
import time
import shutil
import argparse
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# Target shard size to match climbmix format
TARGET_ROWS_PER_SHARD = 86_016  # ~88MB per shard at ~1KB avg doc
TARGET_ROWS_PER_ROW_GROUP = 1024
VAL_SHARD_INDEX = 6542  # Must match prepare.py's VAL_SHARD

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    "fineweb-edu": {
        "description": "FineWeb-Edu: 1.3T tokens of educationally-scored web text (10BT sample)",
        "base_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT",
        "source_files": [f"{i:03d}_00000.parquet" for i in range(14)],
        "text_column": "text",
        "extra_columns": ["score", "int_score"],
        "min_score": None,
        "est_source_size_gb": 2.15,  # per source file
    },
    "fineweb-edu-high": {
        "description": "FineWeb-Edu (score >= 3): highest-quality educational subset",
        "base_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT",
        "source_files": [f"{i:03d}_00000.parquet" for i in range(14)],
        "text_column": "text",
        "extra_columns": ["score", "int_score"],
        "min_score": 3,
        "est_source_size_gb": 2.15,
    },
    "cosmopedia-v2": {
        "description": "Cosmopedia v2: 39M synthetic textbooks/blogposts by Mixtral (from SmolLM-Corpus)",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2",
        "source_files": [f"train-{i:05d}-of-00104.parquet" for i in range(104)],
        "text_column": "text",
        "extra_columns": [],
        "min_score": None,
        "est_source_size_gb": 1.18,
    },
    "slimpajama": {
        "description": "SlimPajama-6B: 6B tokens from deduplicated RedPajama (C4, Wikipedia, Books, ArXiv, Code)",
        "base_url": "https://huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/refs%2Fconvert%2Fparquet/default/train",
        "source_files": [f"{i:04d}.parquet" for i in range(48)],
        "text_column": "text",
        "extra_columns": [],
        "min_score": None,
        "est_source_size_gb": 0.29,
    },
    "python-edu": {
        "description": "Python-Edu: 7.7M educational Python files scored for quality (from SmolLM-Corpus)",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/python-edu",
        "source_files": [f"train-{i:05d}-of-00025.parquet" for i in range(25)],
        "text_column": "text",
        "extra_columns": ["score", "int_score"],
        "min_score": None,
        "est_source_size_gb": 0.4,
    },
}

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url, dest_path, max_attempts=5):
    """Download a file with retries and progress."""
    if os.path.exists(dest_path):
        return True

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            temp_path = dest_path + ".tmp"
            downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = 100 * downloaded / total
                            mb = downloaded / (1024 * 1024)
                            total_mb = total / (1024 * 1024)
                            print(f"\r    {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

            os.rename(temp_path, dest_path)
            print(f"\r    Downloaded {os.path.basename(dest_path)} ({downloaded / (1024*1024):.0f} MB)")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"\n    Attempt {attempt}/{max_attempts} failed: {e}")
            for path in [dest_path + ".tmp", dest_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    return False


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def rechunk_to_shards(source_dir, dataset_config, num_shards, output_dir):
    """
    Read source parquet files, extract text column, and write as
    uniformly-sized shards matching the climbmix format.

    Returns (num_train_shards, num_val_shard) written.
    """
    text_col = dataset_config["text_column"]
    min_score = dataset_config.get("min_score")
    score_col = "int_score" if min_score is not None else None

    # Collect all source files
    source_files = sorted(
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )

    if not source_files:
        print("ERROR: No source parquet files found!")
        return 0

    print(f"\n  Source files: {len(source_files)}")
    if min_score is not None:
        print(f"  Quality filter: score >= {min_score}")

    os.makedirs(output_dir, exist_ok=True)

    # Stream through source files and write fixed-size shards
    shard_idx = 0
    doc_buffer = []
    total_docs = 0
    filtered_docs = 0

    for src_path in source_files:
        fname = os.path.basename(src_path)
        print(f"\n  Processing {fname}...")
        pf = pq.ParquetFile(src_path)

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=[text_col] + ([score_col] if score_col else []))
            texts = rg.column(text_col).to_pylist()
            scores = rg.column(score_col).to_pylist() if score_col else [None] * len(texts)

            for text, score in zip(texts, scores):
                total_docs += 1

                # Quality filter
                if min_score is not None and score is not None and score < min_score:
                    filtered_docs += 1
                    continue

                # Skip empty/tiny docs
                if not text or len(text) < 50:
                    filtered_docs += 1
                    continue

                doc_buffer.append(text)

                # Flush when buffer is full
                if len(doc_buffer) >= TARGET_ROWS_PER_SHARD:
                    _write_shard(output_dir, shard_idx, doc_buffer[:TARGET_ROWS_PER_SHARD])
                    doc_buffer = doc_buffer[TARGET_ROWS_PER_SHARD:]
                    shard_idx += 1

                    if shard_idx >= num_shards + 1:  # +1 for val shard
                        break

            if shard_idx >= num_shards + 1:
                break
        if shard_idx >= num_shards + 1:
            break

        # Progress
        if total_docs > 0:
            kept = total_docs - filtered_docs
            print(f"    Cumulative: {total_docs:,} docs read, {kept:,} kept, {shard_idx} shards written")

    # Write remaining buffer as final shard
    if doc_buffer and shard_idx < num_shards + 1:
        _write_shard(output_dir, shard_idx, doc_buffer)
        shard_idx += 1

    # Rename the last shard to be the validation shard (shard_06542)
    if shard_idx > 0:
        last_shard = shard_idx - 1
        last_path = os.path.join(output_dir, f"shard_{last_shard:05d}.parquet")
        val_path = os.path.join(output_dir, f"shard_{VAL_SHARD_INDEX:05d}.parquet")

        if last_shard != VAL_SHARD_INDEX:
            shutil.move(last_path, val_path)
            print(f"\n  Pinned shard_{last_shard:05d} as validation shard_{VAL_SHARD_INDEX:05d}")

    train_shards = shard_idx - 1  # last one became val
    print(f"\n  Total: {train_shards} training shards + 1 validation shard")
    print(f"  Documents: {total_docs:,} read, {filtered_docs:,} filtered, {total_docs - filtered_docs:,} kept")

    return train_shards


def _write_shard(output_dir, shard_idx, texts):
    """Write a list of texts as a single shard parquet file."""
    path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")

    # Build table with row groups matching climbmix format
    schema = pa.schema([("text", pa.string())])
    writer = pq.ParquetWriter(path, schema)

    for rg_start in range(0, len(texts), TARGET_ROWS_PER_ROW_GROUP):
        rg_texts = texts[rg_start:rg_start + TARGET_ROWS_PER_ROW_GROUP]
        table = pa.table({"text": rg_texts}, schema=schema)
        writer.write_table(table)

    writer.close()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"    Wrote shard_{shard_idx:05d}.parquet ({len(texts):,} docs, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Backup / Restore
# ---------------------------------------------------------------------------

def backup_current(dataset_name):
    """Backup current data + tokenizer before conversion."""
    backup_dir = os.path.join(CACHE_DIR, f"backup_{dataset_name}")

    if os.path.exists(backup_dir):
        print(f"  Backup already exists at {backup_dir}")
        return backup_dir

    os.makedirs(backup_dir, exist_ok=True)

    # Backup data
    if os.path.exists(DATA_DIR):
        data_backup = os.path.join(backup_dir, "data")
        print(f"  Backing up data → {data_backup}")
        shutil.copytree(DATA_DIR, data_backup)

    # Backup tokenizer
    if os.path.exists(TOKENIZER_DIR):
        tok_backup = os.path.join(backup_dir, "tokenizer")
        print(f"  Backing up tokenizer → {tok_backup}")
        shutil.copytree(TOKENIZER_DIR, tok_backup)

    # Record what was backed up
    with open(os.path.join(backup_dir, "source.txt"), "w") as f:
        f.write("climbmix-400b-shuffle\n")

    print(f"  Backup complete: {backup_dir}")
    return backup_dir


def restore_backup(backup_name=None):
    """Restore from a previous backup."""
    if backup_name:
        backup_dir = os.path.join(CACHE_DIR, f"backup_{backup_name}")
    else:
        # Find any backup
        backups = [d for d in os.listdir(CACHE_DIR) if d.startswith("backup_")]
        if not backups:
            print("ERROR: No backups found!")
            return False
        # Prefer the climbmix backup
        climbmix = [b for b in backups if "climbmix" in b or "original" in b]
        backup_name = climbmix[0] if climbmix else backups[0]
        backup_dir = os.path.join(CACHE_DIR, backup_name)

    if not os.path.exists(backup_dir):
        print(f"ERROR: Backup not found at {backup_dir}")
        return False

    print(f"Restoring from {backup_dir}...")

    # Restore data
    data_backup = os.path.join(backup_dir, "data")
    if os.path.exists(data_backup):
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        shutil.copytree(data_backup, DATA_DIR)
        print(f"  Restored data")

    # Restore tokenizer
    tok_backup = os.path.join(backup_dir, "tokenizer")
    if os.path.exists(tok_backup):
        if os.path.exists(TOKENIZER_DIR):
            shutil.rmtree(TOKENIZER_DIR)
        shutil.copytree(tok_backup, TOKENIZER_DIR)
        print(f"  Restored tokenizer")

    print("Restore complete! Ready to train with original dataset.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert alternative datasets for autoresearch experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  fineweb-edu       FineWeb-Edu 10BT sample (all quality levels)
  fineweb-edu-high  FineWeb-Edu 10BT sample (score >= 3 only)
  cosmopedia-v2     Synthetic textbooks/blogposts by Mixtral (39M docs)
  slimpajama        SlimPajama-6B multi-source blend (6B tokens)
  python-edu        Educational Python code files (7.7M files)

Examples:
  uv run convert_dataset.py fineweb-edu
  uv run convert_dataset.py cosmopedia-v2 --num-shards 10 --num-source 3
  uv run convert_dataset.py slimpajama --num-shards 10 --num-source 6
  uv run convert_dataset.py --restore
  uv run convert_dataset.py --list-backups
        """,
    )
    parser.add_argument("dataset", nargs="?", choices=list(DATASETS.keys()),
                        help="Dataset to convert")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of training shards to produce (default: 10)")
    parser.add_argument("--num-source", type=int, default=3,
                        help="Number of source files to download (default: 3, ~6.5GB)")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory for raw downloads (default: ~/.cache/autoresearch/raw_<dataset>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without downloading")
    parser.add_argument("--restore", action="store_true",
                        help="Restore the original climbmix dataset from backup")
    parser.add_argument("--list-backups", action="store_true",
                        help="List available backups")
    parser.add_argument("--skip-backup", action="store_true",
                        help="Skip backing up current dataset (dangerous)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Download workers (default: 4)")

    args = parser.parse_args()

    # --- List backups ---
    if args.list_backups:
        backups = [d for d in os.listdir(CACHE_DIR) if d.startswith("backup_")]
        if not backups:
            print("No backups found.")
        else:
            print("Available backups:")
            for b in sorted(backups):
                path = os.path.join(CACHE_DIR, b)
                source_file = os.path.join(path, "source.txt")
                source = open(source_file).read().strip() if os.path.exists(source_file) else "unknown"
                data_dir = os.path.join(path, "data")
                n_shards = len([f for f in os.listdir(data_dir) if f.endswith(".parquet")]) if os.path.exists(data_dir) else 0
                print(f"  {b}: {source} ({n_shards} shards)")
        return

    # --- Restore ---
    if args.restore:
        # Find the climbmix backup specifically
        backups = [d for d in os.listdir(CACHE_DIR) if d.startswith("backup_")]
        if not backups:
            print("ERROR: No backups found to restore from!")
            sys.exit(1)
        # Use first backup that's from the original dataset
        for b in sorted(backups):
            source_file = os.path.join(CACHE_DIR, b, "source.txt")
            if os.path.exists(source_file):
                source = open(source_file).read().strip()
                if "climbmix" in source:
                    restore_backup(b.replace("backup_", ""))
                    return
        # Fallback: restore any backup
        restore_backup(backups[0].replace("backup_", ""))
        return

    # --- Convert ---
    if not args.dataset:
        parser.print_help()
        sys.exit(1)

    dataset_config = DATASETS[args.dataset]
    print(f"Dataset: {args.dataset}")
    print(f"  {dataset_config['description']}")
    print(f"  Training shards to produce: {args.num_shards}")
    print(f"  Source files to download: {args.num_source}")

    source_files = dataset_config["source_files"][:args.num_source]
    base_url = dataset_config["base_url"]

    # Estimate sizes
    est_source_gb = dataset_config.get("est_source_size_gb", 1.0)
    est_download_gb = args.num_source * est_source_gb
    est_output_gb = (args.num_shards + 1) * 0.088
    print(f"  Estimated download: ~{est_download_gb:.1f} GB")
    print(f"  Estimated output: ~{est_output_gb:.1f} GB")

    if args.dry_run:
        print("\n  DRY RUN — files that would be downloaded:")
        for f in source_files:
            print(f"    {base_url}/{f}")
        print(f"\n  Would produce {args.num_shards} training shards + 1 val shard")
        print(f"  Output: {DATA_DIR}")
        return

    # Step 1: Backup current dataset
    if not args.skip_backup:
        print("\nStep 1: Backing up current dataset...")
        backup_current(args.dataset)
    else:
        print("\nStep 1: Skipping backup (--skip-backup)")

    # Step 2: Download source files
    download_dir = args.download_dir or os.path.join(CACHE_DIR, f"raw_{args.dataset}")
    os.makedirs(download_dir, exist_ok=True)
    print(f"\nStep 2: Downloading source files to {download_dir}")

    for i, filename in enumerate(source_files):
        url = f"{base_url}/{filename}"
        dest = os.path.join(download_dir, filename)
        print(f"\n  [{i+1}/{len(source_files)}] {filename}")
        success = download_file(url, dest)
        if not success:
            print(f"  FATAL: Failed to download {filename}")
            sys.exit(1)

    # Step 3: Clear current data + tokenizer
    print(f"\nStep 3: Clearing current data and tokenizer...")
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"  Removed {DATA_DIR}")
    if os.path.exists(TOKENIZER_DIR):
        shutil.rmtree(TOKENIZER_DIR)
        print(f"  Removed {TOKENIZER_DIR}")

    # Step 4: Convert to shards
    print(f"\nStep 4: Converting to autoresearch shard format...")
    num_written = rechunk_to_shards(download_dir, dataset_config, args.num_shards, DATA_DIR)

    if num_written == 0:
        print("\nERROR: No shards were written! Check the source data.")
        sys.exit(1)

    # Step 5: Verify
    print(f"\nStep 5: Verification...")
    shards = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    val_shard = f"shard_{VAL_SHARD_INDEX:05d}.parquet"
    has_val = val_shard in shards
    train_shards = [s for s in shards if s != val_shard]

    print(f"  Training shards: {len(train_shards)}")
    print(f"  Validation shard: {'OK' if has_val else 'MISSING!'}")

    # Spot-check a shard
    if train_shards:
        check_path = os.path.join(DATA_DIR, train_shards[0])
        pf = pq.ParquetFile(check_path)
        rg = pf.read_row_group(0)
        cols = rg.column_names
        n_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        sample = rg.column("text").to_pylist()[0][:200]
        print(f"  Spot check {train_shards[0]}:")
        print(f"    Columns: {cols}")
        print(f"    Rows: {n_rows:,}")
        print(f"    Sample: {sample}...")

    if not has_val:
        print("\nERROR: Validation shard missing! Something went wrong.")
        sys.exit(1)

    total_size = sum(
        os.path.getsize(os.path.join(DATA_DIR, f))
        for f in shards
    ) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Dataset: {args.dataset}")
    print(f"  Location: {DATA_DIR}")
    print(f"  Shards: {len(train_shards)} train + 1 val ({total_size:.0f} MB total)")
    print(f"")
    print(f"Next steps:")
    print(f"  1. Train tokenizer:  uv run prepare.py")
    print(f"  2. Run experiment:   uv run train.py")
    print(f"  3. Or agent mode:    uv run dashboard.py --agent --tag fineweb-edu")
    print(f"")
    print(f"To restore the original dataset:")
    print(f"  uv run convert_dataset.py --restore")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
