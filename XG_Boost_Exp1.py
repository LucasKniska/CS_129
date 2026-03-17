"""
NBA Win Predictor — Old-Era Generalisation Experiment
══════════════════════════════════════════════════════
  Train    : seasons 2016–2023  (same as nba_win_predictor.py)
  Validate : season  2024       (early stopping — same as nba_win_predictor.py)
  Test     : seasons 2003–2010  (held-out old-era evaluation, touched once)

The dropout mask is kept identical to nba_win_predictor.py so that the two
scripts are directly comparable.  The only addition here is a per-season MAE
line chart for seasons 2003–2010 so you can see whether the model degrades
gracefully as the gap to the training era widens.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "nba_data/final/nba_ml_dataset.csv"
OUTPUT_DIR  = "models"
N_PLAYERS   = 8
RANDOM_SEED = 1

# ── Year-based split boundaries ────────────────────────────────────────────────
YEAR_COL    = "season"
TRAIN_YEARS = range(2016, 2024)   # 2016–2023  →  training        ← SYNCED
VAL_YEARS   = [2024]              # 2024       →  validation       ← SYNCED
TEST_YEARS  = range(2003, 2011)   # 2003–2010  →  held-out old-era test

# ── Column dropout mask  (identical to nba_win_predictor.py) ──────────────────
DROP_PLAYER_COLS: list[str] = [
    # ── Admin / non-predictive ──────────────────────────
    "draft_year", "draft_round", "draft_number", "college", "country",
    # ── Counting totals (raw volume, scale with minutes) ─
    "offensiveRb", "defensiveRb", "fieldGoals", "fieldAttempts",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  {df.shape[0]} team-seasons, {df.shape[1]} columns")

if YEAR_COL not in df.columns:
    raise ValueError(
        f"Year column '{YEAR_COL}' not found. "
        f"Available columns: {list(df.columns)}"
    )

print(f"  Years in file : {sorted(df[YEAR_COL].unique())}")
print(f"  Teams in file : {sorted(df['team'].unique())}")
print(f"  Wins range    : {df['reg_season_wins'].min()} – {df['reg_season_wins'].max()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Drop rows with no player stats
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Apply year-based splits
# ══════════════════════════════════════════════════════════════════════════════
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

unassigned = df_clean[~(train_mask | val_mask | test_mask)][YEAR_COL].unique()
if len(unassigned):
    print(f"\n  ℹ  Years not assigned to any split (ignored): {sorted(unassigned)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Define feature columns
# ══════════════════════════════════════════════════════════════════════════════
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
print(f"  Feature columns: {len(FEAT_COLS)}  (using {N_PLAYERS} player slots)")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Build split arrays
# ══════════════════════════════════════════════════════════════════════════════
X_train = df_clean.loc[train_mask, FEAT_COLS].fillna(0)
y_train = df_clean.loc[train_mask, "reg_season_wins"]

X_val   = df_clean.loc[val_mask,   FEAT_COLS].fillna(0)
y_val   = df_clean.loc[val_mask,   "reg_season_wins"]

X_test  = df_clean.loc[test_mask,  FEAT_COLS].fillna(0)
y_test  = df_clean.loc[test_mask,  "reg_season_wins"]

print(f"\n  Train  : {len(X_train):>3} rows  ({min(TRAIN_YEARS)}–{max(TRAIN_YEARS)})")
print(f"  Val    : {len(X_val):>3} rows  ({VAL_YEARS[0]})")
print(f"  Test   : {len(X_test):>3} rows  ({min(TEST_YEARS)}–{max(TEST_YEARS)})")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Train XGBoost
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def report(label, X, y, df_rows, plot=False):
    """Print MAE/R² and optionally produce an actual-vs-predicted scatter."""
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
        return results

    plot_data = sorted(
        zip(results["team"], results[YEAR_COL],
            results["reg_season_wins"], results["predicted"]),
        key=lambda x: x[3]
    )
    teams     = [f"{d[0]}·{d[1]}" for d in plot_data]
    actual    = np.array([d[2] for d in plot_data])
    predicted = np.array([d[3] for d in plot_data])
    x         = np.arange(len(teams))

    BG, PANEL, GRID = "#0d1117", "#161b22", "#21262d"
    ACT, PRED, TEXT = "#3fb950", "#f85149", "#e6edf3"
    SUB             = "#8b949e"

    fig, ax = plt.subplots(figsize=(max(20, len(teams) * 0.45), 9))
    fig.patch.set_facecolor(BG);  ax.set_facecolor(PANEL)

    for i in range(len(teams)):
        ax.plot([x[i], x[i]], [predicted[i], actual[i]],
                color="white", alpha=0.35, linewidth=1.4,
                linestyle=(0, (3, 3)), zorder=2)

    ax.scatter(x, predicted, color=PRED, s=90, zorder=4,
               edgecolors="white", linewidths=0.6)
    ax.scatter(x, actual,    color=ACT,  s=90, zorder=4,
               edgecolors="white", linewidths=0.6)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.xaxis.grid(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(colors=SUB, length=0)
    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=45, ha="right",
                       fontsize=7, color=TEXT, fontfamily="monospace")
    ax.set_ylabel("Wins", color=TEXT, fontsize=12, labelpad=10)
    ax.yaxis.set_tick_params(labelcolor=SUB, labelsize=10)
    ax.set_xlim(-0.7, len(teams) - 0.3)
    ax.set_ylim(0, max(actual.max(), predicted.max()) + 8)

    fig.text(0.5, 0.97, f"NBA {label.strip()} — Actual vs Expected Wins",
             ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.925,
             f"Sorted by expected wins  ·  MAE: {mae:.2f}  ·  R²: {r2:.3f}",
             ha="center", va="top", fontsize=10, color=SUB)

    pred_h = mlines.Line2D([], [], color=PRED, marker='o', markersize=8,
                           linestyle='None', markeredgecolor='white',
                           markeredgewidth=0.6, label='Expected Wins')
    act_h  = mlines.Line2D([], [], color=ACT,  marker='o', markersize=8,
                           linestyle='None', markeredgecolor='white',
                           markeredgewidth=0.6, label='Actual Wins')
    ax.legend(handles=[pred_h, act_h], loc="upper left", framealpha=0.2,
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    safe = label.strip().replace(" ", "_").replace("/", "-")
    plt.savefig(f"report_{safe}.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.show()
    return results

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Evaluate train / val
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ Evaluation ══════════════════════════════════════════")
report("Train (2016–2023)", X_train, y_train, df_clean[train_mask])
report("Val (2024)",        X_val,   y_val,   df_clean[val_mask], plot=True)

# ══════════════════════════════════════════════════════════════════════════════
# 9.  Old-era test  (touched once — final evaluation only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ⚠  Running old-era test evaluation — do this only once!")
test_results = report("Test (2003–2010)", X_test, y_test,
                      df_clean[test_mask], plot=True)

# ══════════════════════════════════════════════════════════════════════════════
# 10.  Per-season MAE line chart  (2003–2010)
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ Per-Season MAE (old-era test) ════════════════════════")

# Attach predictions back to the test rows so we can group by season
test_preds = model.predict(X_test)
test_rows  = df_clean[test_mask].copy().reset_index(drop=True)
test_rows["predicted"] = test_preds
test_rows["abs_error"] = np.abs(test_rows["predicted"] - test_rows["reg_season_wins"])

season_mae = (
    test_rows.groupby(YEAR_COL)["abs_error"]
    .mean()
    .reset_index()
    .rename(columns={"abs_error": "mae"})
    .sort_values(YEAR_COL)
)

# Also compute season-level R² for the subtitle annotation
season_r2 = (
    test_rows.groupby(YEAR_COL)
    .apply(lambda g: r2_score(g["reg_season_wins"], g["predicted"]))
    .reset_index()
    .rename(columns={0: "r2"})
    .sort_values(YEAR_COL)
)

season_stats = season_mae.merge(season_r2, on=YEAR_COL)
print(f"\n  {'Season':<8} {'MAE':>6} {'R²':>7} {'Teams':>6}")
print("  " + "─" * 32)
for _, row in season_stats.iterrows():
    n_teams = (test_rows[YEAR_COL] == row[YEAR_COL]).sum()
    print(f"  {int(row[YEAR_COL]):<8} {row['mae']:>6.2f} {row['r2']:>7.3f} {n_teams:>6}")

overall_mae = season_mae["mae"].mean()
print(f"\n  Overall old-era MAE (mean of season MAEs): {overall_mae:.2f} wins")

# ── Line chart ────────────────────────────────────────────────────────────────
seasons  = season_stats[YEAR_COL].astype(int).tolist()
mae_vals = season_stats["mae"].tolist()
r2_vals  = season_stats["r2"].tolist()

# Colour constants — white background, readable
LINE_C   = "#1565C0"   # blue line / markers
FILL_C   = "#BBDEFB"   # light-blue fill under curve
AVG_C    = "#C62828"   # red dashed average line
GRID_C   = "#E0E0E0"
TEXT_C   = "#212121"
SUB_C    = "#616161"

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Shaded area under the MAE curve
ax.fill_between(seasons, mae_vals, alpha=0.18, color=FILL_C, zorder=1)

# MAE line + markers
ax.plot(seasons, mae_vals,
        color=LINE_C, linewidth=2.5, marker="o",
        markersize=9, markerfacecolor=LINE_C,
        markeredgecolor="white", markeredgewidth=1.5,
        zorder=3, label="Per-season MAE")

# Annotate each point: MAE value above, R² below
for sx, my, ry in zip(seasons, mae_vals, r2_vals):
    ax.annotate(f"{my:.2f}",
                xy=(sx, my), xytext=(0, 11),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=LINE_C)
    ax.annotate(f"R²={ry:.2f}",
                xy=(sx, my), xytext=(0, -16),
                textcoords="offset points",
                ha="center", va="top",
                fontsize=8, color=SUB_C)

# Overall-average reference line
ax.axhline(overall_mae, color=AVG_C, linewidth=1.6,
           linestyle="--", zorder=2,
           label=f"Mean MAE = {overall_mae:.2f} wins")

# Grid + spines
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID_C, linewidth=0.9, linestyle="--")
ax.xaxis.grid(False)
for side, spine in ax.spines.items():
    spine.set_visible(side in ("bottom", "left"))
    if side in ("bottom", "left"):
        spine.set_color(GRID_C)

ax.tick_params(axis="both", length=0, labelcolor=TEXT_C)
ax.set_xticks(seasons)
ax.set_xticklabels([str(s) for s in seasons], fontsize=10, color=TEXT_C)
ax.set_yticks(np.arange(0, max(mae_vals) + 3, 1))
ax.yaxis.set_tick_params(labelsize=10, labelcolor=SUB_C)

# Y-axis lower bound: always 0 so the scale isn't misleadingly zoomed
ax.set_ylim(0, max(mae_vals) + 3.5)
ax.set_xlim(min(seasons) - 0.4, max(seasons) + 0.4)

ax.set_xlabel("Season", fontsize=12, color=TEXT_C, labelpad=8)
ax.set_ylabel("Mean Absolute Error (wins)", fontsize=12, color=TEXT_C, labelpad=8)
ax.set_title("Per-Season MAE — Old-Era Test (2003–2010)\n"
             "Model trained on 2016–2023, validated on 2024",
             fontsize=14, fontweight="bold", color=TEXT_C, pad=12)

ax.legend(loc="upper right", frameon=True, framealpha=0.9,
          facecolor="white", edgecolor=GRID_C,
          labelcolor=TEXT_C, fontsize=10)

plt.tight_layout()
plt.savefig("report_old_era_mae_by_season.png", dpi=160,
            bbox_inches="tight", facecolor="white")
plt.show()
print("  Saved: report_old_era_mae_by_season.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11.  Feature importance
# ══════════════════════════════════════════════════════════════════════════════
importances = pd.Series(model.feature_importances_, index=FEAT_COLS)
top20 = importances.nlargest(20)
print("\n══ Top 20 Features ══════════════════════════════════════")
for feat, imp in top20.items():
    print(f"  {feat:<40} {imp:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 12.  Save model artifacts
# ══════════════════════════════════════════════════════════════════════════════
model_path    = os.path.join(OUTPUT_DIR, "xgb_model_old_era.pkl")
features_path = os.path.join(OUTPUT_DIR, "feature_cols_old_era.json")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(features_path, "w") as f:
    json.dump(FEAT_COLS, f, indent=2)

print(f"\n✅ Saved:")
print(f"   {model_path}")
print(f"   {features_path}")