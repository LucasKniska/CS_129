"""
NBA ML Dataset Collector
=========================
Builds an ML-ready dataset for predicting team wins from player combinations.

Data sources:
  - https://api.server.nbaapi.com  — player totals + advanced stats (BBRef-sourced)
  - https://site.api.espn.com      — standings (regular season wins/losses)

Y (targets):
    reg_season_wins, reg_losses

X (features):
    - Top 10 players by minutes played per team-season
      Each player: all fields returned by playeradvancedstats + playertotals
      No hardcoded feature list — uses whatever the API returns
    - Team-level aggregates computed from the top-10 player pool

Install:
    pip install requests pandas numpy

Usage:
    python nba_ml_collector.py

Output:
    nba_data/
        raw/        - Cached JSON per season (standings + season rows)
        final/
            nba_ml_dataset.csv
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import json
import traceback
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SEASONS_BACK   = 10    # How many seasons to collect
CURRENT_SEASON = 2025  # Most recent season end-year (2025 = 2024-25)
TOP_N_PLAYERS  = 10    # Players per row — sorted by minutesPlayed descending
MIN_GAMES      = 10    # Minimum games played (filters 10-day contracts)
OUTPUT_DIR     = "nba_data"
RATE_LIMIT_SEC = 1.0   # Seconds between API requests
# ─────────────────────────────────────────────────────────────

RAW_DIR   = os.path.join(OUTPUT_DIR, "raw")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")
for d in [RAW_DIR, FINAL_DIR]:
    os.makedirs(d, exist_ok=True)

SEASONS = list(range(CURRENT_SEASON - SEASONS_BACK + 1, CURRENT_SEASON + 1))

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

# ESPN team abbreviations → ESPN team IDs (as used in standings API)
ESPN_TEAM_IDS = {
    'ATL': '1',  'BOS': '2',  'BKN': '17', 'CHA': '30', 'CHI': '4',
    'CLE': '5',  'DAL': '6',  'DEN': '7',  'DET': '8',  'GS':  '9',
    'HOU': '10', 'IND': '11', 'LAC': '12', 'LAL': '13', 'MEM': '29',
    'MIA': '14', 'MIL': '15', 'MIN': '16', 'NO': '3',  'NY':  '18',
    'OKC': '25', 'ORL': '19', 'PHI': '20', 'PHX': '21', 'POR': '22',
    'SAC': '23', 'SA':  '24', 'TOR': '28', 'UTAH': '26','WSH': '27',
}

# ESPN abbrev → nbaapi.com/BBRef abbrev (used for player stat API calls)
ESPN_TO_NBAAPI = {
    'GS':   'GSW',
    'NY':   'NYK',
    'SA':   'SAS',
    'UTAH': 'UTA',
    'WSH':  'WAS',
    'BKN':  'BRK',
    'CHA':  'CHO',
    'PHX':  'PHO',
    'NO':   'NOP',
}

# nbaapi.com abbrev → ESPN abbrev (used when looking up standings)
NBAAPI_TO_ESPN = {v: k for k, v in ESPN_TO_NBAAPI.items()}

ALL_TEAMS = list(ESPN_TEAM_IDS.keys())

# Columns to drop before embedding — internal keys not useful as ML features
DROP_COLS = {'id', 'playerId', '_team', '_season', 'isPlayoff', 'season', 'team'}


# ═══════════════════════════════════════════════════════════════
#  HTTP HELPER
# ═══════════════════════════════════════════════════════════════

def safe_get(url: str, params: dict = None, retries: int = 3) -> dict | None:
    """GET with retry — mirrors the pattern from your ESPN win prob scraper."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"      Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(RATE_LIMIT_SEC)
            return resp.json()
        except Exception as e:
            wait = RATE_LIMIT_SEC * (attempt + 2)
            print(f"      Attempt {attempt+1} failed ({e}) — retrying in {wait:.0f}s")
            time.sleep(wait)
    print(f"      All retries exhausted: {url}")
    return None


# ═══════════════════════════════════════════════════════════════
#  SOURCE 1: nbaapi.com — player stats
# ═══════════════════════════════════════════════════════════════

def get_player_totals(team: str, season: int) -> pd.DataFrame:
    """
    Per-game totals for all players on a team-season.
    Fields: playerName, position, age, games, gamesStarted, minutesPg,
            fieldGoals, fieldAttempts, fieldPercent, threeFg, threeAttempts,
            threePercent, twoFg, twoAttempts, twoPercent, effectFgPercent,
            ft, ftAttempts, ftPercent, offensiveRb, defensiveRb, totalRb,
            assists, steals, blocks, turnovers, personalFouls, points
    """
    nbaapi_team = ESPN_TO_NBAAPI.get(team, team)
    data = safe_get(
        "https://api.server.nbaapi.com/api/playertotals",
        params={'season': season, 'team': nbaapi_team, 'pageSize': 30, 'page': 1}
    )
    if not data or 'data' not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    df['_team']   = team
    df['_season'] = season
    return df


def get_player_advanced(team: str, season: int) -> pd.DataFrame:
    """
    Advanced stats for all players on a team-season.
    Fields: playerName, position, age, games, minutesPlayed, per, tsPercent,
            threePAR, ftr, offensiveRBPercent, defensiveRBPercent,
            totalRBPercent, assistPercent, stealPercent, blockPercent,
            turnoverPercent, usagePercent, offensiveWS, defensiveWS,
            winShares, winSharesPer, obpm, dbpm, bpm, vorp
    """
    nbaapi_team = ESPN_TO_NBAAPI.get(team, team)
    data = safe_get(
        "https://api.server.nbaapi.com/api/playeradvancedstats",
        params={'season': season, 'team': nbaapi_team, 'pageSize': 30, 'page': 1}
    )
    if not data or 'data' not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    df['_team']   = team
    df['_season'] = season
    return df


# ═══════════════════════════════════════════════════════════════
#  SOURCE 2: ESPN — regular season standings
# ═══════════════════════════════════════════════════════════════

def get_espn_standings(season: int) -> dict:
    """
    Returns {team_abbrev: {'wins': int, 'losses': int}} for all 30 teams.
    Cached to disk after first fetch.
    """
    cache_path = os.path.join(RAW_DIR, f"espn_standings_{season}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    data = safe_get(
        "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
        params={'season': season}
    )

    standings = {}
    if not data:
        return standings

    try:
        for conference in data.get('children', []):
            for entry in conference.get('standings', {}).get('entries', []):
                abbrev = entry.get('team', {}).get('abbreviation', '').upper()
                wins = losses = 0
                for stat in entry.get('stats', []):
                    if stat.get('name') == 'wins':
                        wins = int(stat.get('value', 0))
                    elif stat.get('name') == 'losses':
                        losses = int(stat.get('value', 0))
                if abbrev:
                    standings[abbrev] = {'wins': wins, 'losses': losses}
    except Exception as e:
        print(f"    [WARN] standings parse error season {season}: {e}")

    with open(cache_path, 'w') as f:
        json.dump(standings, f)

    return standings


# ═══════════════════════════════════════════════════════════════
#  CLEANING & MERGING
# ═══════════════════════════════════════════════════════════════

def filter_min_games(df: pd.DataFrame) -> pd.DataFrame:
    """Drop players with fewer than MIN_GAMES (removes 10-day contracts)."""
    if df.empty:
        return df
    games_col = next((c for c in df.columns if c.lower() in ('games', 'g', 'gp')), None)
    if games_col:
        df = df.copy()
        df[games_col] = pd.to_numeric(df[games_col], errors='coerce')
        df = df[df[games_col] >= MIN_GAMES]
    return df.reset_index(drop=True)


def merge_totals_and_advanced(tot: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """
    Outer join totals + advanced on playerName + _team + _season.
    Drops duplicate non-key columns from advanced to avoid _x/_y suffixes.
    """
    if tot.empty and adv.empty:
        return pd.DataFrame()
    if tot.empty:
        return adv
    if adv.empty:
        return tot

    join_keys = [k for k in ['playerName', '_team', '_season']
                 if k in tot.columns and k in adv.columns]

    overlap = (set(tot.columns) & set(adv.columns)) - set(join_keys)
    adv_slim = adv.drop(columns=list(overlap), errors='ignore')

    return tot.merge(adv_slim, on=join_keys, how='outer')


def sort_and_trim(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by minutesPlayed descending, keep top TOP_N_PLAYERS.
    Falls back to minutesPg if minutesPlayed not present.
    """
    if df.empty:
        return df

    minutes_col = next(
        (c for c in df.columns if c.lower() in ('minutesplayed', 'minutespg', 'mp', 'min')),
        None
    )
    if minutes_col:
        df = df.copy()
        df[minutes_col] = pd.to_numeric(df[minutes_col], errors='coerce')
        df = df.sort_values(minutes_col, ascending=False)

    return df.head(TOP_N_PLAYERS).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  ML ROW BUILDER
# ═══════════════════════════════════════════════════════════════

def safe_float(val) -> float:
    try:
        f = float(val)
        return np.nan if (np.isinf(f) or np.isnan(f)) else round(f, 4)
    except (TypeError, ValueError):
        return np.nan


def gini(arr: np.ndarray) -> float:
    """Usage concentration — high = star-driven offense, low = balanced."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan
    arr = np.sort(arr.astype(float))
    n   = len(arr)
    idx = np.arange(1, n + 1)
    denom = n * arr.sum()
    return float((2 * (idx * arr).sum() - (n + 1) * arr.sum()) / denom) if denom else np.nan


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Return all columns from the merged player dataframe that are
    useful ML features — everything except internal/identifier cols.
    """
    return [c for c in df.columns
            if c not in DROP_COLS and c != 'playerName']


def build_ml_row(team: str, season: int,
                 player_df: pd.DataFrame,
                 reg_wins: int, reg_losses: int) -> dict:
    """
    Flatten one team-season into a single ML-ready dict.

    Structure:
      season, team                         — metadata
      reg_season_wins, reg_losses          — targets
      team_*                               — team aggregates
      p1_name, p1_<feat>, p1_<feat>...    — top player by minutes
      p2_name, p2_<feat>, ...             — second player
      ...up to p{TOP_N_PLAYERS}
    """
    row = {
        'season':          season,
        'team':            team,
        'reg_season_wins': reg_wins   if reg_wins   >= 0 else np.nan,
        'reg_losses':      reg_losses if reg_losses >= 0 else np.nan,
    }

    if player_df.empty:
        return row

    # Determine feature columns from actual API response
    feature_cols = get_feature_cols(player_df)

    # ── Embed each player slot ──
    for i in range(1, TOP_N_PLAYERS + 1):
        slot = f'p{i}'
        if i <= len(player_df):
            p = player_df.iloc[i - 1]
            row[f'{slot}_name'] = p.get('playerName', np.nan)
            for col in feature_cols:
                row[f'{slot}_{col}'] = safe_float(p.get(col, np.nan))
        else:
            # Empty slot — fewer than TOP_N_PLAYERS qualified players
            row[f'{slot}_name'] = np.nan
            for col in feature_cols:
                row[f'{slot}_{col}'] = np.nan

    # ── Team aggregate features ──
    def col_series(name):
        return pd.to_numeric(player_df.get(name, pd.Series(dtype=float)), errors='coerce')

    row['team_total_ws']          = safe_float(col_series('winShares').sum())
    row['team_avg_per']           = safe_float(col_series('per').mean())
    row['team_avg_bpm']           = safe_float(col_series('bpm').mean())
    row['team_total_vorp']        = safe_float(col_series('vorp').sum())
    row['team_max_usg']           = safe_float(col_series('usagePercent').max())
    row['team_usg_gini']          = safe_float(gini(col_series('usagePercent').values))
    row['team_players_qualified'] = len(player_df)

    return row


# ═══════════════════════════════════════════════════════════════
#  PER-TEAM PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_team_season(team: str, season: int, standings: dict) -> dict:
    print(f"    [{team}] ", end='', flush=True)

    # Fetch both stat tables
    tot_df = get_player_totals(team, season)
    adv_df = get_player_advanced(team, season)

    # Merge → filter 10-day → sort by minutes → top 10
    players = merge_totals_and_advanced(tot_df, adv_df)
    players = filter_min_games(players)
    players = sort_and_trim(players)

    # Standings are keyed by ESPN abbrev — convert if needed
    espn_key  = NBAAPI_TO_ESPN.get(team, team)
    standing  = standings.get(espn_key, standings.get(team, {}))
    reg_wins   = int(standing.get('wins',   -1))
    reg_losses = int(standing.get('losses', -1))

    row = build_ml_row(team, season, players, reg_wins, reg_losses)
    print(f"{reg_wins}W / {reg_losses}L  |  {len(players)} players ✓")
    return row


# ═══════════════════════════════════════════════════════════════
#  SEASON ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def collect_season(season: int) -> list:
    cache_path = os.path.join(RAW_DIR, f"season_{season}_rows.json")
    if os.path.exists(cache_path):
        print(f"  Season {season}: loading from cache")
        with open(cache_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Season {season-1}-{str(season)[-2:]}  |  {len(ALL_TEAMS)} teams")
    print(f"{'='*60}")

    print(f"  Fetching ESPN standings...")
    standings = get_espn_standings(season)
    print(f"  Got standings for {len(standings)} teams")

    rows = []
    for team in ALL_TEAMS:
        try:
            row = process_team_season(team, season, standings)
            rows.append(row)
        except Exception as e:
            print(f"ERROR — {e}")
            traceback.print_exc()
            rows.append({'season': season, 'team': team, 'error': str(e)})

    with open(cache_path, 'w') as f:
        json.dump(rows, f, default=str)

    return rows


# ═══════════════════════════════════════════════════════════════
#  DATASET ASSEMBLY
# ═══════════════════════════════════════════════════════════════

def build_dataset():
    print("\n" + "="*60)
    print("  NBA ML Dataset Builder")
    print(f"  Seasons : {SEASONS[0]}-{SEASONS[-1]}  ({len(SEASONS)} seasons)")
    print(f"  Teams   : {len(ALL_TEAMS)}")
    print(f"  Top-N   : {TOP_N_PLAYERS} players per row (by minutes played)")
    print(f"  Min games: {MIN_GAMES}")
    print("="*60)

    all_rows = []
    for season in SEASONS:
        rows = collect_season(season)
        all_rows.extend(rows)
        # Incremental save after each season
        pd.DataFrame(all_rows).to_csv(
            os.path.join(RAW_DIR, "interim_dataset.csv"), index=False
        )

    df = pd.DataFrame(all_rows)

    # ── Column ordering: meta → targets → team → players ──
    meta_cols   = ['season', 'team']
    target_cols = ['reg_season_wins', 'reg_losses']
    team_cols   = sorted([c for c in df.columns if c.startswith('team_')])
    player_cols = [c for c in df.columns
                   if any(c.startswith(f'p{i}_') for i in range(1, TOP_N_PLAYERS + 1))]
    # Sort player cols so p1_* all appear together, then p2_*, etc.
    player_cols = sorted(player_cols, key=lambda c: (int(c.split('_')[0][1:]), c))
    other_cols  = [c for c in df.columns
                   if c not in meta_cols + target_cols + team_cols + player_cols]

    ordered = meta_cols + target_cols + team_cols + player_cols + other_cols
    df = df[[c for c in ordered if c in df.columns]]

    out_path = os.path.join(FINAL_DIR, "nba_ml_dataset.csv")
    df.to_csv(out_path, index=False)
    return df, out_path


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, path: str):
    print(f"\n{'='*60}")
    print("  COMPLETE")
    print(f"{'='*60}")
    print(f"  File      : {path}")
    print(f"  Rows      : {len(df)}")
    print(f"  Columns   : {len(df.columns)}")

    print(f"\n  Targets:")
    for col in ['reg_season_wins', 'reg_losses']:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                print(f"    {col:<22} mean={s.mean():.1f}  "
                      f"min={s.min():.0f}  max={s.max():.0f}  n={len(s)}")

    print(f"\n  Player 1 feature columns:")
    for c in sorted([c for c in df.columns if c.startswith('p1_')]):
        print(f"    {c}")

    print(f"\n  Team aggregate columns:")
    for c in sorted([c for c in df.columns if c.startswith('team_')]):
        print(f"    {c}")

    print(f"\n  Seasons : {sorted(df['season'].dropna().unique().tolist())}")
    print(f"{'='*60}")
    print(f"\n  Load with: pd.read_csv('{path}')")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = datetime.now()
    print(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    df, path = build_dataset()
    print_summary(df, path)
    print(f"\nElapsed: {datetime.now() - t0}")