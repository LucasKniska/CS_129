"""
NBA Win Predictor - XGBoost trained on modern era, tested on old era
  Train    : seasons 2017-2025  (model trains on these)
  Validate : seasons 2015-2016  (early stopping / hyperparam tuning)
  Test     : seasons 2003-2009  (held-out old-era evaluation, touched once)

Year column assumed to be named 'year' in the CSV (e.g. 2015 = 2014-15 season).
Change YEAR_COL below if your column has a different name.
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
N_PLAYERS   = 8
RANDOM_SEED = 1

# ── Year-based split boundaries ────────────────────────────────────────────────
YEAR_COL    = "season"          # column in CSV that holds the season year
TRAIN_YEARS = range(2017, 2026)   # 2017-2025  →  training
VAL_YEARS   = range(2015, 2017)   # 2015-2016  →  validation / early stopping
TEST_YEARS  = range(2003, 2010)   # 2003-2009  →  held-out old-era test

# ── Column dropout mask ────────────────────────────────────────────────────────
DROP_PLAYER_COLS: list[str] = [
    # ── Admin / non-predictive ──────────────────────────
    "draft_year",
    "draft_round",
    "draft_number",
    "college",
    "country",

    # ── Playing time (leakage) ──────────────────────────
    "games",
    "gamesStarted",
    "minutesPlayed",
    "minutesPg",

    # ── Counting totals (scale with minutes, leak) ──────
    "assists",
    "blocks",
    "steals",
    "points",
    "totalRb",
    "offensiveRb",
    "defensiveRb",
    "fieldGoals",
    "fieldAttempts",
    "ft",
    "ftAttempts",
    "threeAttempts",
    "threeFg",
    "twoAttempts",
    "twoFg",
    "turnovers",
    "personalFouls",

    # ── Win-derived / outcome stats (heavy leakage) ─────
    "winShares",
    "winSharesPer",
    "box",
    "offensiveBox",
    "defensiveBox",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  {df.shape[0]} team-seasons, {df.shape[1]} columns")

# Validate year column exists
if YEAR_COL not in df.columns:
    raise ValueError(
        f"Year column '{YEAR_COL}' not found in CSV. "
        f"Available columns: {list(df.columns)}"
    )

print(f"  Years in file : {sorted(df[YEAR_COL].unique())}")
print(f"  Teams in file : {sorted(df['team'].unique())}")
print(f"  Wins range    : {df['reg_season_wins'].min()} – {df['reg_season_wins'].max()}")

# ── Detect and remove rows with no player stats ────────────────────────────────
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
                     .groupby(YEAR_COL)["team"]
                     .apply(list)
                     .reset_index())
    for _, row in empty_summary.iterrows():
        teams   = row["team"]
        preview = ", ".join(teams[:5]) + ("..." if len(teams) > 5 else "")
        print(f"   Year {int(row[YEAR_COL])}: {len(teams)} teams ({preview})")
else:
    print("\n  No empty rows found ✓")

df_clean = df[complete_mask].copy()
print(f"\n  Usable rows: {len(df_clean)} (dropped {n_empty})")

# ── Apply year-based splits ────────────────────────────────────────────────────
train_mask = df_clean[YEAR_COL].isin(TRAIN_YEARS)
val_mask   = df_clean[YEAR_COL].isin(VAL_YEARS)
test_mask  = df_clean[YEAR_COL].isin(TEST_YEARS)

print("\n── Year-based splits ─────────────────────────────────────")
for label, mask, years in [
    ("Train",    train_mask, TRAIN_YEARS),
    ("Validate", val_mask,   VAL_YEARS),
    ("Test",     test_mask,  TEST_YEARS),
]:
    count  = mask.sum()
    yr_str = f"{min(years)}–{max(years)}"
    status = f"{count:>3} rows ✓" if count > 0 else "⚠  NO ROWS — check YEAR_COL name/values"
    print(f"  {label:<9}: years {yr_str}  →  {status}")

# Warn about any years in the CSV that fall outside all three splits
all_assigned = train_mask | val_mask | test_mask
unassigned   = df_clean[~all_assigned][YEAR_COL].unique()
if len(unassigned):
    print(f"\n  ℹ  Years not assigned to any split (ignored): {sorted(unassigned)}")
    print(f"     These are 2010–2014 seasons sitting between eras — intentionally excluded.")

# ── Define feature columns ─────────────────────────────────────────────────────
NAME_COLS = [c for c in df.columns if c.endswith("_name")]
META_COLS = [
    YEAR_COL, "split", "team", "reg_season_wins", "reg_losses",
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

DROPPED_BY_MASK = set()
if DROP_PLAYER_COLS:
    for suffix in DROP_PLAYER_COLS:
        for i in range(1, N_PLAYERS + 1):
            DROPPED_BY_MASK.add(f"p{i}_{suffix}")
    print(f"\n  Dropping {len(DROPPED_BY_MASK)} columns via dropout mask.")

FEAT_COLS = [
    c for c in df_clean.columns
    if c not in EXCLUDE
    and c not in DROPPED_BY_MASK
    and df_clean[c].dtype in ["float64", "int64"]
    and (c not in ALL_PLAYER_SLOTS or c in ALLOWED_SLOTS)
]
print(f"\n  Feature columns: {len(FEAT_COLS)}  (using {N_PLAYERS} player slots)")

# ── Build splits ───────────────────────────────────────────────────────────────
X_train = df_clean.loc[train_mask, FEAT_COLS].fillna(0)
y_train = df_clean.loc[train_mask, "reg_season_wins"]

X_val   = df_clean.loc[val_mask,   FEAT_COLS].fillna(0)
y_val   = df_clean.loc[val_mask,   "reg_season_wins"]

X_test  = df_clean.loc[test_mask,  FEAT_COLS].fillna(0)
y_test  = df_clean.loc[test_mask,  "reg_season_wins"]

print(f"\n  Train  : {len(X_train):>3} rows  ({min(TRAIN_YEARS)}–{max(TRAIN_YEARS)})")
print(f"  Val    : {len(X_val):>3} rows  ({min(VAL_YEARS)}–{max(VAL_YEARS)})")
print(f"  Test   : {len(X_test):>3} rows  ({min(TEST_YEARS)}–{max(TEST_YEARS)})")

# ── Train XGBoost ──────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")

model = XGBRegressor(
    random_state=RANDOM_SEED,
    min_child_weight=15,
    max_depth=2,
    reg_lambda=8,
    reg_alpha=2,
    subsample=0.7,
    colsample_bytree=0.4,
    learning_rate=0.005,
    n_estimators=10000,
    early_stopping_rounds=300,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100,
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def report(label, X, y, df_rows, plot=False):
    preds = model.predict(X)
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"\n  ── {label} ──────────────────────────────────────")
    print(f"     MAE : {mae:.2f} wins")
    print(f"     R²  : {r2:.3f}")

    results = df_rows[[YEAR_COL, "team", "reg_season_wins"]].copy()
    results["predicted"] = np.round(preds, 1)
    results["error"]     = np.round(preds - y.values, 1)

    if not plot:
        return

    plot_data = sorted(
        zip(results["team"], results[YEAR_COL],
            results["reg_season_wins"], results["predicted"]),
        key=lambda x: x[3]
    )
    # Label as "TEAM·YEAR" so duplicate teams across seasons are distinguishable
    teams     = [f"{d[0]}·{d[1]}" for d in plot_data]
    actual    = np.array([d[2] for d in plot_data])
    predicted = np.array([d[3] for d in plot_data])
    x         = np.arange(len(teams))

    BG_COLOR      = "#0d1117"
    PANEL_COLOR   = "#161b22"
    GRID_COLOR    = "#21262d"
    ACTUAL_COLOR  = "#3fb950"
    PREDICT_COLOR = "#f85149"
    TEXT_COLOR    = "#e6edf3"
    SUBTEXT_COLOR = "#8b949e"

    fig, ax = plt.subplots(figsize=(max(20, len(teams) * 0.45), 9))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    for i in range(len(teams)):
        ax.plot([x[i], x[i]], [predicted[i], actual[i]],
                color="white", alpha=0.35, linewidth=1.4,
                linestyle=(0, (3, 3)), zorder=2)

    ax.scatter(x, predicted, color=PREDICT_COLOR, s=90, zorder=4,
               edgecolors="white", linewidths=0.6)
    ax.scatter(x, actual,    color=ACTUAL_COLOR,  s=90, zorder=4,
               edgecolors="white", linewidths=0.6)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=SUBTEXT_COLOR, length=0)

    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=45, ha="right",
                       fontsize=7, color=TEXT_COLOR, fontfamily="monospace")
    ax.set_ylabel("Wins", color=TEXT_COLOR, fontsize=12, labelpad=10)
    ax.yaxis.set_tick_params(labelcolor=SUBTEXT_COLOR, labelsize=10)
    ax.set_xlim(-0.7, len(teams) - 0.3)
    ax.set_ylim(0, max(actual.max(), predicted.max()) + 8)

    fig.text(0.5, 0.97, f"NBA {label.strip()} — Actual vs Expected Wins",
             ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.925,
             f"Sorted by expected wins  ·  MAE: {mae:.2f}  ·  R²: {r2:.3f}",
             ha="center", va="top", fontsize=10, color=SUBTEXT_COLOR)

    pred_handle = mlines.Line2D([], [], color=PREDICT_COLOR, marker='o',
                                markersize=8, linestyle='None',
                                markeredgecolor='white', markeredgewidth=0.6,
                                label='Expected Wins')
    act_handle  = mlines.Line2D([], [], color=ACTUAL_COLOR, marker='o',
                                markersize=8, linestyle='None',
                                markeredgecolor='white', markeredgewidth=0.6,
                                label='Actual Wins')
    ax.legend(handles=[pred_handle, act_handle], loc="upper left",
              framealpha=0.2, facecolor=PANEL_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    safe_label = label.strip().replace(" ", "_").replace("/", "-")
    plt.savefig(f"report_{safe_label}.png", dpi=160,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()

print("\n══ Evaluation ══════════════════════════════════════════")
report("Train (2017–2025)", X_train, y_train, df_clean[train_mask])
report("Val   (2015–2016)", X_val,   y_val,   df_clean[val_mask], plot=True)

# ── Old-era test — uncomment when you're satisfied with val ───────────────────
print("\n  ⚠  Running old-era test evaluation — do this only once!")
report("Test (2003–2009)", X_test, y_test, df_clean[test_mask], plot=True)

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