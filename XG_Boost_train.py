"""
NBA Win Predictor - XGBoost Training Pipeline
Run: python train_model.py
Outputs: models/xgb_model.pkl, models/feature_cols.json

Splits:
  Train    : 2016-2023  (model trains on these)
  Validate : 2024       (early stopping / hyperparam tuning)
  Test     : 2025       (final held-out evaluation, touched once)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "nba_data/final/nba_ml_dataset.csv"
OUTPUT_DIR  = "models"
N_PLAYERS   = 5
RANDOM_SEED = 1

TRAIN_SEASONS = list(range(2016, 2024))   # 2016–2023 inclusive
VAL_SEASONS   = [2024]
TEST_SEASONS  = [2025]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  {df.shape[0]} team-seasons, {df.shape[1]} columns")
print(f"  Seasons in file: {sorted(df['season'].unique())}")
print(f"  Wins range: {df['reg_season_wins'].min()} – {df['reg_season_wins'].max()}")

# ── Detect and remove rows with no player stats ────────────────────────────────
# Some seasons (e.g. 2018) have wins/losses but all player columns are null.
# These rows are useless for training and would corrupt the model — an all-zero
# feature vector would be mapped to a real win total with no signal.
stat_cols_check = [
    c for c in df.columns
    if any(c.startswith(f"p{i}_") for i in range(1, N_PLAYERS + 1))
    and not c.endswith("_name")
]

empty_mask    = df[stat_cols_check].isnull().all(axis=1)
complete_mask = ~empty_mask
n_empty       = empty_mask.sum()

if n_empty > 0:
    print(f"\n⚠  {n_empty} rows have no player stats — excluded from all splits:")
    empty_summary = (df[empty_mask]
                     .groupby("season")["team"]
                     .apply(list)
                     .reset_index())
    for _, row in empty_summary.iterrows():
        teams = row["team"]
        preview = ", ".join(teams[:5]) + ("..." if len(teams) > 5 else "")
        print(f"   Season {int(row['season'])}: {len(teams)} teams ({preview})")
else:
    print("\n  No empty rows found ✓")

df_clean = df[complete_mask].copy()
print(f"\n  Usable rows: {len(df_clean)} (dropped {n_empty})")

# ── Validate all target seasons are present after cleaning ─────────────────────
for label, seasons in [("Train", TRAIN_SEASONS),
                        ("Val",   VAL_SEASONS),
                        ("Test",  TEST_SEASONS)]:
    found   = [s for s in seasons if s in df_clean["season"].values]
    missing = [s for s in seasons if s not in df_clean["season"].values]
    status  = "⚠  missing: " + str(missing) if missing else "✓"
    print(f"  {label:<6}: seasons {found}  {status}")

# ── Define feature columns ─────────────────────────────────────────────────────
NAME_COLS = [c for c in df.columns if c.endswith("_name")]
META_COLS = [
    "season", "team", "reg_season_wins", "reg_losses",
    # Team-level aggregates are derived from player stats — excluding prevents
    # leakage and forces the model to learn from raw individual player data
    "team_avg_bpm", "team_avg_per", "team_max_usg",
    "team_players_qualified", "team_total_vorp",
    "team_total_ws", "team_usg_gini",
]
EXCLUDE = set(META_COLS + NAME_COLS)

# Player slot columns that exist in the CSV (p1_ through p10_)
ALL_PLAYER_SLOTS = set(
    c for c in df_clean.columns
    if any(c.startswith(f"p{i}_") for i in range(1, 11))
)
# Only keep slots 1..N_PLAYERS; drop higher slots entirely
ALLOWED_SLOTS = set(
    c for c in ALL_PLAYER_SLOTS
    if any(c.startswith(f"p{i}_") for i in range(1, N_PLAYERS + 1))
)

FEAT_COLS = [
    c for c in df_clean.columns
    if c not in EXCLUDE
    and df_clean[c].dtype in ["float64", "int64"]
    and (c not in ALL_PLAYER_SLOTS or c in ALLOWED_SLOTS)
]
print(f"\n  Feature columns: {len(FEAT_COLS)}  (using {N_PLAYERS} player slots)")

# ── Build splits ───────────────────────────────────────────────────────────────
train_mask = df_clean["season"].isin(TRAIN_SEASONS)
val_mask   = df_clean["season"].isin(VAL_SEASONS)
test_mask  = df_clean["season"].isin(TEST_SEASONS)

X_train = df_clean.loc[train_mask, FEAT_COLS].fillna(0)
y_train = df_clean.loc[train_mask, "reg_season_wins"]

X_val   = df_clean.loc[val_mask,   FEAT_COLS].fillna(0)
y_val   = df_clean.loc[val_mask,   "reg_season_wins"]

X_test  = df_clean.loc[test_mask,  FEAT_COLS].fillna(0)
y_test  = df_clean.loc[test_mask,  "reg_season_wins"]

print(f"\n  Train  : {len(X_train):>3} rows  (seasons {TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]})")
print(f"  Val    : {len(X_val):>3} rows  (season  {VAL_SEASONS})")
print(f"  Test   : {len(X_test):>3} rows  (season  {TEST_SEASONS})")

# ── Train XGBoost ──────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")

model = XGBRegressor(
    random_state = RANDOM_SEED,
    min_child_weight=8,
    reg_lambda=2,
    max_depth=3,
    learning_rate=.01,
    n_estimators=1000,
    early_stopping_rounds=100,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100,
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
def report(label, X, y, df_rows):
    preds   = model.predict(X)
    mae     = mean_absolute_error(y, preds)
    r2      = r2_score(y, preds)
    print(f"\n  ── {label} ──────────────────────────────────────")
    print(f"     MAE : {mae:.2f} wins")
    print(f"     R²  : {r2:.3f}")
    results = df_rows[["season", "team", "reg_season_wins"]].copy()
    results["predicted"] = np.round(preds, 1)
    results["error"]     = np.round(preds - y.values, 1)
    print(results.sort_values("error", key=abs, ascending=False)
                 .to_string(index=False))

print("\n══ Evaluation ══════════════════════════════════════════")
report("Train (2016-2023)", X_train, y_train, df_clean[train_mask])
report("Val   (2024)",      X_val,   y_val,   df_clean[val_mask])

# Test — only run once you're satisfied with val performance
# if len(X_test) > 0:
#     print("\n  ⚠  Running test evaluation — do this only once!")
#     report("Test  (2025)", X_test, y_test, df_clean[test_mask])
# else:
#     print("\n  Test set (2025) not yet available — skipping.")

# ── Feature importance ─────────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEAT_COLS)
top20 = importances.nlargest(20)
print("\n══ Top 20 Features ══════════════════════════════════════")
for feat, imp in top20.items():
    print(f"  {feat:<40} {imp:.4f}")

# ── Save artifacts ─────────────────────────────────────────────────────────────
model_path    = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
features_path = os.path.join(OUTPUT_DIR, "feature_cols.json")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(features_path, "w") as f:
    json.dump(FEAT_COLS, f, indent=2)

print(f"\n✅ Saved:")
print(f"   {model_path}")
print(f"   {features_path}")