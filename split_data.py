"""
Split NBA ML Dataset into Train / Validation / Test
=====================================================
Reads nba_data/final/nba_ml_dataset.csv and adds a 'split' column:
    1 = train      (80%)
    2 = validation (10%)
    3 = test       (10%)

Split is stratified by season so each split contains a proportional
representation of all seasons rather than a simple random cut.

The original file is overwritten in place with the new 'split' column
inserted as the third column (after season, team).

Usage:
    python split_dataset.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ----------------------------------------------------------------
#  CONFIGURATION
# ----------------------------------------------------------------
DATASET_PATH = os.path.join("nba_data", "final", "nba_ml_dataset.csv")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42

# Split labels
TRAIN = 1
VAL   = 2
TEST  = 3
# ----------------------------------------------------------------


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified split by season.
    Within each season, rows are shuffled then assigned 80/10/10.
    Returns df with a new 'split' column.
    """
    df = df.copy()
    df["split"] = -1

    rng = np.random.default_rng(RANDOM_SEED)

    for season, group in df.groupby("season"):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n = len(idx)

        n_train = round(n * TRAIN_RATIO)
        n_val   = round(n * VAL_RATIO)
        # test gets the remainder to avoid rounding drift
        n_test  = n - n_train - n_val

        df.loc[idx[:n_train],                   "split"] = TRAIN
        df.loc[idx[n_train:n_train + n_val],    "split"] = VAL
        df.loc[idx[n_train + n_val:],           "split"] = TEST

    return df


def reorder_split_col(df: pd.DataFrame) -> pd.DataFrame:
    """Move 'split' to be the 3rd column, after season and team."""
    cols = list(df.columns)
    cols.remove("split")
    insert_at = min(2, len(cols))
    cols.insert(insert_at, "split")
    return df[cols]


def main():
    print("\n" + "="*60)
    print("  NBA ML Dataset — Train/Val/Test Split")
    print(f"  File: {DATASET_PATH}")
    print(f"  Ratio: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")
    print(f"  Seed: {RANDOM_SEED}")
    print("="*60)

    if not os.path.exists(DATASET_PATH):
        print(f"\n  ERROR: {DATASET_PATH} not found.")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"\n  Loaded: {len(df)} rows x {len(df.columns)} cols")

    if "split" in df.columns:
        print("  [INFO] 'split' column already exists — overwriting.")
        df = df.drop(columns=["split"])

    df = assign_splits(df)
    df = reorder_split_col(df)

    # Summary
    counts = df["split"].value_counts().sort_index()
    total  = len(df)
    print(f"\n  Split breakdown:")
    for label, name in [(TRAIN, "train"), (VAL, "validation"), (TEST, "test")]:
        n = counts.get(label, 0)
        print(f"    {label} ({name:<10}) : {n:>5} rows  ({n/total*100:.1f}%)")

    # Per-season breakdown
    print(f"\n  Per-season split counts:")
    season_summary = df.groupby(["season", "split"]).size().unstack(fill_value=0)
    season_summary.columns = ["train", "val", "test"]
    print(season_summary.to_string())

    df.to_csv(DATASET_PATH, index=False)
    print(f"\n  Saved -> {DATASET_PATH}")
    print(f"  {len(df)} rows x {len(df.columns)} cols")
    print("="*60)


if __name__ == "__main__":
    t0 = datetime.now()
    print(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print(f"\nElapsed: {datetime.now() - t0}")