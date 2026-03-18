"""
NBA Win Predictor - Linear Regression Training Pipeline

Splits:
  Train    : 2016-2023
  Validate : 2024
  Test     : 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import os

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "nba_data/final/nba_ml_dataset.csv"
OUTPUT_DIR  = "models"
N_PLAYERS   = 5
RANDOM_SEED = 1

TRAIN_SEASONS = list(range(2016, 2024))
VAL_SEASONS   = [2024]
TEST_SEASONS  = [2025]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)

print(f"  {df.shape[0]} team-seasons, {df.shape[1]} columns")
print(f"  Seasons in file: {sorted(df['season'].unique())}")
print(f"  Wins range: {df['reg_season_wins'].min()} – {df['reg_season_wins'].max()}")

# ── Remove rows with no player stats ───────────────────────────────────────────
stat_cols_check = [
    c for c in df.columns
    if any(c.startswith(f"p{i}_") for i in range(1, N_PLAYERS + 1))
    and not c.endswith("_name")
]

empty_mask    = df[stat_cols_check].isnull().all(axis=1)
complete_mask = ~empty_mask
n_empty       = empty_mask.sum()

if n_empty > 0:
    print(f"\n⚠  {n_empty} rows have no player stats — excluded")

df_clean = df[complete_mask].copy()
print(f"  Usable rows: {len(df_clean)}")

# ── Define feature columns ─────────────────────────────────────────────────────
NAME_COLS = [c for c in df.columns if c.endswith("_name")]

META_COLS = [
    "season", "team", "reg_season_wins", "reg_losses",
    "team_avg_bpm", "team_avg_per", "team_max_usg",
    "team_players_qualified", "team_total_vorp",
    "team_total_ws", "team_usg_gini",
]

EXCLUDE = set(META_COLS + NAME_COLS)

ALL_PLAYER_SLOTS = set(
    c for c in df_clean.columns
    if any(c.startswith(f"p{i}_") for i in range(1, 11))
)

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

print(f"\n  Feature columns: {len(FEAT_COLS)}")

# ── Build splits ───────────────────────────────────────────────────────────────
train_mask = df_clean["season"].isin(TRAIN_SEASONS)
val_mask   = df_clean["season"].isin(VAL_SEASONS)
test_mask  = df_clean["season"].isin(TEST_SEASONS)

X_train = df_clean.loc[train_mask, FEAT_COLS].fillna(0)
y_train = df_clean.loc[train_mask, "reg_season_wins"]

X_val   = df_clean.loc[val_mask, FEAT_COLS].fillna(0)
y_val   = df_clean.loc[val_mask, "reg_season_wins"]

X_test  = df_clean.loc[test_mask, FEAT_COLS].fillna(0)
y_test  = df_clean.loc[test_mask, "reg_season_wins"]

print(f"\nTrain rows: {len(X_train)}")
print(f"Val rows:   {len(X_val)}")
print(f"Test rows:  {len(X_test)}")

# ── Model Pipeline ─────────────────────────────────────────────────────────────
print("\nTraining Linear Regression...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# ── Evaluation function ────────────────────────────────────────────────────────
def report(label, X, y, df_rows):
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    r2  = r2_score(y, preds)

    print(f"\n── {label} ─────────────────────────────")
    print(f"MAE : {mae:.2f} wins")
    print(f"R²  : {r2:.3f}")

    results = df_rows[["season", "team", "reg_season_wins"]].copy()
    results["predicted"] = np.round(preds, 1)
    results["error"]     = np.round(preds - y.values, 1)

    print(results.sort_values("error", key=abs, ascending=False)
                 .to_string(index=False))

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\n══ Evaluation ══════════════════════════════════════════")

report("Train (2016-2023)", X_train, y_train, df_clean[train_mask])
report("Val   (2024)",      X_val,   y_val,   df_clean[val_mask])

# Predictions for validation set
val_preds = model.predict(X_val)

plt.figure(figsize=(6,6))
plt.scatter(y_val, val_preds)

plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()],
         linestyle="--")

plt.xlabel("Actual Wins")
plt.ylabel("Predicted Wins")
plt.title("Predicted vs Actual Wins (Validation)")
plt.show()

# Uncomment later when ready
# report("Test  (2025)", X_test, y_test, df_clean[test_mask])

# ── Coefficient Importance ─────────────────────────────────────────────────────
print("\n══ Top Features (Linear Coefficients) ═════════════════")

reg = model.named_steps["regressor"]
coef = pd.Series(reg.coef_, index=FEAT_COLS)

top20 = coef.abs().nlargest(20)

for feat in top20.index:
    print(f"{feat:<40} {coef[feat]:.4f}")

# ── Save artifacts ─────────────────────────────────────────────────────────────
model_path = os.path.join(OUTPUT_DIR, "linear_model.pkl")
features_path = os.path.join(OUTPUT_DIR, "feature_cols_linear.json")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(features_path, "w") as f:
    json.dump(FEAT_COLS, f, indent=2)

print("\nSaved:")
print(model_path)
print(features_path)

