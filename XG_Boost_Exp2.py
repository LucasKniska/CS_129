"""
NBA Win Predictor — XGBoost
  Train      : seasons 2016–2023  (model trains on these)
  Validate   : season  2024       (early stopping / hyperparam tuning)
  Predict    : season  2025 rosters  +  season 2024 per-player stats
               (2024-25 current roster composition, last year's proven stats)
  Compare    : nba_data/final/2025_espn_predicted_wins.csv

How the 2025 prediction rows are built
---------------------------------------
  1.  Pull every row where season == 2025.  These rows hold the *current*
      2024-25 roster (player names in p1_name … p8_name).
  2.  For each player slot on each team, look up that player's stats from
      season == 2024 in the same CSV.
  3.  Overwrite the stat columns (everything except *_name and meta cols)
      in the 2025 row with those 2024 values.
  4.  If a player has no 2024 row (rookie / new arrival), their stats stay
      as whatever is already in the 2025 row — or zero-filled if absent.
  This approach avoids in-season leakage while still using the real
  2024-25 roster composition.
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
DATA_PATH    = "nba_data/final/nba_ml_dataset.csv"
ESPN_PATH    = "nba_data/final/2025_espn_predicted_wins.csv"
OUTPUT_DIR   = "models"
N_PLAYERS    = 8
RANDOM_SEED  = 1

# ── Year-based split boundaries ────────────────────────────────────────────────
YEAR_COL     = "season"
TRAIN_YEARS  = range(2016, 2024)   # 2016–2023  →  training
VAL_YEARS    = [2024]              # 2024       →  validation / early stopping
PRED_YEAR    = 2025                # roster year  (stats swapped from 2024)
STAT_YEAR    = 2024                # stat source year for the prediction rows

# ── Column dropout mask ────────────────────────────────────────────────────────
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
# 2.  Build the 2025 prediction rows
#     Roster = season 2025 rows  |  Stats = swapped from each player's 2024 row
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nBuilding {PRED_YEAR} prediction rows using {STAT_YEAR} player stats...")

name_cols_all  = [c for c in df.columns if c.endswith("_name")]
player_slots   = [f"p{i}" for i in range(1, N_PLAYERS + 1)]

# Identify stat columns per slot (everything pN_* that isn't pN_name)
def slot_stat_cols(slot: str) -> list[str]:
    return [c for c in df.columns
            if c.startswith(f"{slot}_") and not c.endswith("_name")]

# Build two lookups from STAT_YEAR rows, scanning ALL slots (p1–p10):
#   player_stat_lookup : name → {stat_suffix: value}
#   player_team_lookup : name → [team1, team2, ...]  (in CSV row order)
# ─────────────────────────────────────────────────────────────────────────────
# Traded players appear once per team they played for.  The CSV is assumed
# to be ordered so their destination/second-team row comes AFTER their
# origin-team row.  We always overwrite on each new occurrence, meaning
# player_stat_lookup ends up holding the LAST (destination) team's stats —
# which is exactly what we want for projecting forward into next season.
# player_team_lookup records every team they appeared on so we can log
# which stint was actually used.
# ─────────────────────────────────────────────────────────────────────────────
stat_year_rows = df[df[YEAR_COL] == STAT_YEAR].copy()

# Detect every slot present in the CSV (could be p1–p10 or more)
all_csv_slots = sorted({
    c.split("_")[0]
    for c in df.columns
    if c.startswith("p") and "_" in c and c.split("_")[0][1:].isdigit()
}, key=lambda s: int(s[1:]))

player_stat_lookup: dict[str, dict[str, float]] = {}
player_team_lookup: dict[str, list[str]]        = {}

for slot in all_csv_slots:
    name_col  = f"{slot}_name"
    stat_cols = slot_stat_cols(slot)
    if name_col not in stat_year_rows.columns or not stat_cols:
        continue
    sub = stat_year_rows[["team", name_col] + stat_cols].dropna(subset=[name_col])
    for _, row in sub.iterrows():
        pname = row[name_col]
        if pd.isna(pname):
            continue
        stats_this_row = {
            col.replace(f"{slot}_", ""): row[col]
            for col in stat_cols
        }
        # Always overwrite → last row = destination/second team for traded players
        player_stat_lookup[pname] = stats_this_row
        player_team_lookup.setdefault(pname, [])
        team_val = str(row.get("team", ""))
        if team_val and team_val not in player_team_lookup[pname]:
            player_team_lookup[pname].append(team_val)

print(f"  Players found in {STAT_YEAR} data: {len(player_stat_lookup)}")

# Now build the 2025 prediction DataFrame
pred_rows_raw = df[df[YEAR_COL] == PRED_YEAR].copy()
if len(pred_rows_raw) == 0:
    raise ValueError(
        f"No rows found for season {PRED_YEAR} in {DATA_PATH}. "
        "Make sure the CSV contains 2025 roster rows."
    )

print(f"  {len(pred_rows_raw)} teams found in {PRED_YEAR} roster data")

pred_rows = pred_rows_raw.copy()
swapped_count   = 0
missing_players = []

for slot in player_slots:
    name_col  = f"{slot}_name"
    stat_cols = slot_stat_cols(slot)
    if name_col not in pred_rows.columns:
        continue
    for idx, row in pred_rows.iterrows():
        pname = row.get(name_col)
        if pd.isna(pname):
            continue
        if pname in player_stat_lookup:
            stats = player_stat_lookup[pname]
            for col in stat_cols:
                suffix = col.replace(f"{slot}_", "")
                if suffix in stats:
                    pred_rows.at[idx, col] = stats[suffix]
            swapped_count += 1
        else:
            missing_players.append((row.get("team", "?"), slot, pname))

print(f"  Stat swaps performed : {swapped_count}")

# Report traded players (appeared on >1 team in STAT_YEAR) — shows which stint was used
traded = {n: teams for n, teams in player_team_lookup.items() if len(teams) > 1}
if traded:
    print(f"  Traded players in {STAT_YEAR} (using LAST/destination team stats): {len(traded)}")
    for name, teams in list(traded.items())[:10]:
        print(f"    {name}: {' -> '.join(teams)}")
    if len(traded) > 10:
        print(f"    ... and {len(traded) - 10} more")

if missing_players:
    print(f"  Players with no {STAT_YEAR} stats (rookie/overseas, zero-filled): "
          f"{len(missing_players)}")
    for team, slot, name in missing_players[:10]:
        print(f"    {team} {slot}: {name}")
    if len(missing_players) > 10:
        print(f"    ... and {len(missing_players) - 10} more")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Drop rows with no player stats; build clean working set
# ══════════════════════════════════════════════════════════════════════════════
stat_cols_check = [
    c for c in df.columns
    if any(c.startswith(f"p{i}_") for i in range(1, N_PLAYERS + 1))
    and not c.endswith("_name")
]

# Remove empty rows from historical data only (not pred rows — those may be partial)
empty_mask    = df[df[YEAR_COL] != PRED_YEAR][stat_cols_check].isnull().all(axis=1)
complete_mask = ~empty_mask
n_empty       = empty_mask.sum()

df_hist  = df[df[YEAR_COL] != PRED_YEAR][complete_mask].copy()
df_clean = pd.concat([df_hist, pred_rows], ignore_index=True)

if n_empty:
    print(f"\n⚠  {n_empty} historical rows with no player stats excluded.")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Apply splits
# ══════════════════════════════════════════════════════════════════════════════
train_mask = df_clean[YEAR_COL].isin(TRAIN_YEARS)
val_mask   = df_clean[YEAR_COL].isin(VAL_YEARS)
pred_mask  = df_clean[YEAR_COL] == PRED_YEAR

print("\n── Year-based splits ─────────────────────────────────────")
for label, mask, yr in [
    ("Train",    train_mask, f"{min(TRAIN_YEARS)}–{max(TRAIN_YEARS)}"),
    ("Validate", val_mask,   str(VAL_YEARS[0])),
    ("Predict",  pred_mask,  str(PRED_YEAR)),
]:
    count  = mask.sum()
    status = f"{count:>3} rows ✓" if count > 0 else "⚠  NO ROWS"
    print(f"  {label:<9}: years {yr}  →  {status}")

unassigned = df_clean[~(train_mask | val_mask | pred_mask)][YEAR_COL].unique()
if len(unassigned):
    print(f"\n  ℹ  Years not assigned to any split (ignored): {sorted(unassigned)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Define feature columns
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
# 6.  Build split arrays
# ══════════════════════════════════════════════════════════════════════════════
X_train = df_clean.loc[train_mask, FEAT_COLS].fillna(0)
y_train = df_clean.loc[train_mask, "reg_season_wins"]

X_val   = df_clean.loc[val_mask,   FEAT_COLS].fillna(0)
y_val   = df_clean.loc[val_mask,   "reg_season_wins"]

X_pred  = df_clean.loc[pred_mask,  FEAT_COLS].fillna(0)
# No y_pred ground truth yet — that's the point!

print(f"\n  Train  : {len(X_train):>3} rows  ({min(TRAIN_YEARS)}–{max(TRAIN_YEARS)})")
print(f"  Val    : {len(X_val):>3} rows  ({VAL_YEARS[0]})")
print(f"  Predict: {len(X_pred):>3} rows  ({PRED_YEAR}, no ground truth)")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Train XGBoost
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
# 8.  Evaluate on train / val
# ══════════════════════════════════════════════════════════════════════════════

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

    BG, PANEL, GRID   = "#0d1117", "#161b22", "#21262d"
    ACT, PRED, TEXT   = "#3fb950", "#f85149", "#e6edf3"
    SUB               = "#8b949e"

    fig, ax = plt.subplots(figsize=(max(20, len(teams) * 0.45), 9))
    fig.patch.set_facecolor(BG);  ax.set_facecolor(PANEL)

    for i in range(len(teams)):
        ax.plot([x[i], x[i]], [predicted[i], actual[i]],
                color="white", alpha=0.35, linewidth=1.4,
                linestyle=(0, (3, 3)), zorder=2)
    ax.scatter(x, predicted, color=PRED, s=90, zorder=4, edgecolors="white", linewidths=0.6)
    ax.scatter(x, actual,    color=ACT,  s=90, zorder=4, edgecolors="white", linewidths=0.6)

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
    fig.text(0.5, 0.925, f"Sorted by expected wins  ·  MAE: {mae:.2f}  ·  R²: {r2:.3f}",
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

print("\n══ Evaluation ══════════════════════════════════════════")
report("Train (2016–2023)", X_train, y_train, df_clean[train_mask])
report("Val (2024)",        X_val,   y_val,   df_clean[val_mask], plot=True)

# ══════════════════════════════════════════════════════════════════════════════
# 9.  Predict 2024-25 season  +  compare to ESPN  +  actual wins
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n══ 2024-25 Predictions (roster={PRED_YEAR}, stats={STAT_YEAR}) ══════════")

pred_wins = model.predict(X_pred)

# Pull model predictions + actual wins that are already in the 2025 rows
pred_df = df_clean.loc[pred_mask, ["team", "reg_season_wins"]].copy().reset_index(drop=True)
pred_df.rename(columns={"reg_season_wins": "actual_wins"}, inplace=True)
pred_df["model_pred_wins"]   = np.round(pred_wins, 1)
pred_df["model_pred_losses"] = np.round(82 - pred_wins, 1)

# ── Load ESPN predictions ──────────────────────────────────────────────────────
espn_df = pd.read_csv(ESPN_PATH)
espn_df.columns = espn_df.columns.str.strip()

compare = pred_df.merge(
    espn_df[["team", "espn_pred_wins", "espn_pred_losses"]],
    on="team", how="left"
)

# Error columns vs actual
compare["model_err"] = np.round(compare["model_pred_wins"] - compare["actual_wins"], 1)
compare["espn_err"]  = np.round(compare["espn_pred_wins"]  - compare["actual_wins"], 1)

# Summary MAE figures
has_actual = compare["actual_wins"].notna()
if has_actual.sum() > 0:
    model_mae = np.abs(compare.loc[has_actual, "model_err"]).mean()
    espn_mae  = np.abs(compare.loc[has_actual & compare["espn_pred_wins"].notna(), "espn_err"]).mean()
    print(f"\n  Model MAE vs actual : {model_mae:.2f} wins")
    print(f"  ESPN  MAE vs actual : {espn_mae:.2f} wins")

compare = compare.sort_values("actual_wins", ascending=False).reset_index(drop=True)

print(f"\n{'Team':<6} {'Actual':>7} {'Model':>7} {'ESPN':>7} {'Mod Err':>8} {'ESPN Err':>9}")
print("─" * 48)
for _, row in compare.iterrows():
    actual = f"{row['actual_wins']:.0f}"       if pd.notna(row['actual_wins'])    else "N/A"
    model  = f"{row['model_pred_wins']:.1f}"
    espn   = f"{row['espn_pred_wins']:.1f}"    if pd.notna(row['espn_pred_wins']) else "N/A"
    merr   = f"{row['model_err']:+.1f}"        if pd.notna(row['model_err'])      else "N/A"
    eerr   = f"{row['espn_err']:+.1f}"         if pd.notna(row['espn_err'])       else "N/A"
    print(f"  {row['team']:<6} {actual:>7} {model:>7} {espn:>7} {merr:>8} {eerr:>9}")

# ── Three-way scatter-with-lines plot ─────────────────────────────────────────
# White background.  Teams sorted by actual wins on the x-axis.
# Each team column has three dots stacked vertically:
#   ● Blue   — actual wins (ground truth, drawn largest so it's always visible)
#   ● Purple — model expected wins
#   ● Red    — ESPN predicted wins
# A solid line connects each prediction dot to the actual dot so the error
# direction (over / under) is immediately obvious.
# ─────────────────────────────────────────────────────────────────────────────

plot_df = compare.dropna(subset=["actual_wins"]).copy()
plot_df = plot_df.sort_values("actual_wins").reset_index(drop=True)

teams_p  = plot_df["team"].tolist()
actual_w = plot_df["actual_wins"].values.astype(float)
model_w  = plot_df["model_pred_wins"].values.astype(float)
espn_w   = plot_df["espn_pred_wins"].values.astype(float)   # may contain NaN
x        = np.arange(len(teams_p))

# ── Colour palette (white-background friendly) ─────────────────────────────
ACT_C  = "#1565C0"   # blue   — actual wins
MOD_C  = "#7B1FA2"   # purple — model expected wins
ESPN_C = "#C62828"   # red    — ESPN prediction
GRID_C = "#E0E0E0"   # light grey grid lines
TEXT_C = "#212121"   # near-black labels
SUB_C  = "#616161"   # grey subtitle / tick labels

fig, ax = plt.subplots(figsize=(max(22, len(teams_p) * 0.78), 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Vertical connecting lines (drawn first, behind the dots) ───────────────
for i in range(len(teams_p)):
    # Model → actual
    ax.plot([x[i], x[i]], [model_w[i], actual_w[i]],
            color=MOD_C, alpha=0.5, linewidth=1.8, zorder=2)
    # ESPN → actual  (skip if no ESPN data for this team)
    if not np.isnan(espn_w[i]):
        ax.plot([x[i], x[i]], [espn_w[i], actual_w[i]],
                color=ESPN_C, alpha=0.5, linewidth=1.8, zorder=2)

# ── Scatter dots (draw actual last so it's always on top) ─────────────────
espn_valid = ~np.isnan(espn_w)
if espn_valid.any():
    ax.scatter(x[espn_valid], espn_w[espn_valid],
               color=ESPN_C, s=90, zorder=4,
               edgecolors="white", linewidths=0.8, label="ESPN prediction")

ax.scatter(x, model_w,
           color=MOD_C, s=90, zorder=5,
           edgecolors="white", linewidths=0.8, label="Model expected wins")

ax.scatter(x, actual_w,
           color=ACT_C, s=120, zorder=6,
           edgecolors="white", linewidths=1.0, label="Actual wins")

# ── Axes styling ───────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID_C, linewidth=0.9, linestyle="--")
ax.xaxis.grid(False)

# Keep only bottom and left spines; colour them grey
for side, spine in ax.spines.items():
    if side in ("top", "right"):
        spine.set_visible(False)
    else:
        spine.set_color(GRID_C)

ax.tick_params(axis="both", which="both", length=0, labelcolor=TEXT_C)
ax.set_xticks(x)
ax.set_xticklabels(teams_p, rotation=45, ha="right",
                   fontsize=9, color=TEXT_C, fontfamily="monospace")
ax.set_ylabel("Number of Wins", color=TEXT_C, fontsize=13, labelpad=10)
ax.yaxis.set_tick_params(labelsize=10, labelcolor=SUB_C)
ax.set_xlim(-0.7, len(teams_p) - 0.3)
y_max = max(np.nanmax(actual_w), np.nanmax(model_w),
            np.nanmax(espn_w[espn_valid]) if espn_valid.any() else 0)
ax.set_ylim(0, y_max + 10)

# ── Titles ─────────────────────────────────────────────────────────────────
mae_str = (f"Model MAE: {model_mae:.2f} wins  |  ESPN MAE: {espn_mae:.2f} wins"
           if has_actual.sum() > 0 else "")
ax.set_title("2024-25 NBA Win Predictions vs Actual Results",
             fontsize=17, fontweight="bold", color=TEXT_C, pad=14)
ax.set_xlabel(f"Team  ·  Sorted by actual wins  ·  {mae_str}",
              fontsize=9, color=SUB_C, labelpad=8)

# ── Legend ─────────────────────────────────────────────────────────────────
ax.legend(loc="upper left", frameon=True, framealpha=0.9,
          facecolor="white", edgecolor=GRID_C,
          labelcolor=TEXT_C, fontsize=10, markerscale=1.2)

plt.tight_layout()
plt.savefig("report_2025_three_way.png", dpi=160,
            bbox_inches="tight", facecolor="white")
plt.show()

# ── Save comparison CSV ────────────────────────────────────────────────────────
compare.to_csv("2025_predictions_vs_espn.csv", index=False)
print("\n  Saved: 2025_predictions_vs_espn.csv")
print("  Saved: report_2025_three_way.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Save model artifacts
# ══════════════════════════════════════════════════════════════════════════════
model_path    = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
features_path = os.path.join(OUTPUT_DIR, "feature_cols.json")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(features_path, "w") as f:
    json.dump(FEAT_COLS, f, indent=2)

print(f"\n✅ Saved:")
print(f"   {model_path}")
print(f"   {features_path}")