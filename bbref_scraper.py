"""
NBA ML Dataset Collector — Step 1: Player Stats
================================================
Collects player totals + advanced stats for all team-seasons and saves to:
    nba_data/final/nba_ml_dataset.csv

Run this first, then run nba_physical_collector.py to append physical attributes.

Install:
    pip install requests pandas numpy

Usage:
    python nba_stats_collector.py
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
FIRST_SEASON   = 2018  # Adjust down if probe confirms earlier seasons work
CURRENT_SEASON = 2018
TOP_N_PLAYERS  = 10
MIN_GAMES      = 10
OUTPUT_DIR     = "nba_data"
RATE_LIMIT_SEC = 1.0
# ─────────────────────────────────────────────────────────────

RAW_DIR   = os.path.join(OUTPUT_DIR, "raw")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")
for d in [RAW_DIR, FINAL_DIR]:
    os.makedirs(d, exist_ok=True)

SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

ESPN_TEAM_IDS = {
    'ATL': '1',  'BOS': '2',  'BKN': '17', 'CHA': '30', 'CHI': '4',
    'CLE': '5',  'DAL': '6',  'DEN': '7',  'DET': '8',  'GS':  '9',
    'HOU': '10', 'IND': '11', 'LAC': '12', 'LAL': '13', 'MEM': '29',
    'MIA': '14', 'MIL': '15', 'MIN': '16', 'NO':  '3',  'NY':  '18',
    'OKC': '25', 'ORL': '19', 'PHI': '20', 'PHX': '21', 'POR': '22',
    'SAC': '23', 'SA':  '24', 'TOR': '28', 'UTAH': '26','WSH': '27',
}

HISTORICAL_ABBREVS = {
    'BKN':  [(2013, 9999, 'BRK'), (2001, 2012, 'NJN')],
    'CHA':  [(2015, 9999, 'CHO'), (2005, 2014, 'CHA'), (2001, 2002, 'CHH')],
    'NO':   [(2014, 9999, 'NOP'), (2008, 2013, 'NOH'), (2006, 2007, 'NOK'), (2003, 2005, 'NOH')],
    'MEM':  [(2002, 9999, 'MEM'), (2001, 2001, 'VAN')],
    'OKC':  [(2009, 9999, 'OKC'), (2001, 2008, 'SEA')],
    'WSH':  [(2001, 9999, 'WAS')],
    'GS':   [(2001, 9999, 'GSW')],
    'NY':   [(2001, 9999, 'NYK')],
    'SA':   [(2001, 9999, 'SAS')],
    'UTAH': [(2001, 9999, 'UTA')],
    'PHX':  [(2001, 9999, 'PHO')],
}

ESPN_TO_NBAAPI_CURRENT = {
    'GS': 'GSW', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA',
    'WSH': 'WAS', 'BKN': 'BRK', 'CHA': 'CHO', 'PHX': 'PHO', 'NO': 'NOP',
    'NJN': 'NJ'
}

DROP_COLS = {'id', 'playerId', '_team', '_season', 'isPlayoff', 'season', 'team'}


def espn_to_bbref(espn_abbrev: str, season: int) -> str:
    if espn_abbrev in HISTORICAL_ABBREVS:
        for (first, last, bbref) in HISTORICAL_ABBREVS[espn_abbrev]:
            if first <= season <= last:
                return bbref
    return ESPN_TO_NBAAPI_CURRENT.get(espn_abbrev, espn_abbrev)


def get_active_teams(season: int) -> list:
    active = []
    for espn in ESPN_TEAM_IDS:
        if espn == 'CHA' and season == 2004:
            continue
        if espn == 'NO' and season < 2003:
            continue
        active.append(espn)
    return active


# ═══════════════════════════════════════════════════════════════
#  HTTP
# ═══════════════════════════════════════════════════════════════

def safe_get(url: str, params: dict = None, retries: int = 3):
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
#  PLAYER STATS
# ═══════════════════════════════════════════════════════════════

def get_player_totals(espn_team: str, season: int) -> pd.DataFrame:
    bbref_team = espn_to_bbref(espn_team, season)
    data = safe_get(
        "https://api.server.nbaapi.com/api/playertotals",
        params={'season': season, 'team': bbref_team, 'pageSize': 50, 'page': 1}
    )
    if not data or not data.get('data'):
        return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    df['_team']   = espn_team
    df['_season'] = season
    return df


def get_player_advanced(espn_team: str, season: int) -> pd.DataFrame:
    bbref_team = espn_to_bbref(espn_team, season)
    data = safe_get(
        "https://api.server.nbaapi.com/api/playeradvancedstats",
        params={'season': season, 'team': bbref_team, 'pageSize': 50, 'page': 1}
    )
    if not data or not data.get('data'):
        return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    df['_team']   = espn_team
    df['_season'] = season
    return df


# ═══════════════════════════════════════════════════════════════
#  STANDINGS
# ═══════════════════════════════════════════════════════════════

def get_espn_standings(season: int) -> dict:
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


def lookup_standing(espn_team: str, season: int, standings: dict) -> tuple:
    overrides = {
        'NO':   ['NO', 'NOP', 'NOH', 'NOK'],
        'GS':   ['GS', 'GSW'],
        'NY':   ['NY', 'NYK'],
        'SA':   ['SA', 'SAS'],
        'UTAH': ['UTAH', 'UTA'],
        'WSH':  ['WSH', 'WAS'],
        'BKN':  ['BKN', 'BRK', 'NJN'],
        'CHA':  ['CHA', 'CHO', 'CHH'],
        'MEM':  ['MEM', 'VAN'],
        'OKC':  ['OKC', 'SEA'],
        'PHX':  ['PHX', 'PHO'],
    }
    candidates = list(dict.fromkeys(
        overrides.get(espn_team, [espn_team]) +
        [espn_team, espn_to_bbref(espn_team, season)]
    ))
    for key in candidates:
        if key in standings:
            r = standings[key]
            return int(r.get('wins', -1)), int(r.get('losses', -1))
    return -1, -1


# ═══════════════════════════════════════════════════════════════
#  CLEANING
# ═══════════════════════════════════════════════════════════════

def filter_min_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    games_col = next((c for c in df.columns if c.lower() in ('games', 'g', 'gp')), None)
    if games_col:
        df = df.copy()
        df[games_col] = pd.to_numeric(df[games_col], errors='coerce')
        df = df[df[games_col] >= MIN_GAMES]
    return df.reset_index(drop=True)


def merge_totals_and_advanced(tot: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
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
#  ROW BUILDER
# ═══════════════════════════════════════════════════════════════

def safe_float(val) -> float:
    try:
        f = float(val)
        return np.nan if (np.isinf(f) or np.isnan(f)) else round(f, 4)
    except (TypeError, ValueError):
        return np.nan


def gini(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan
    arr = np.sort(arr.astype(float))
    n   = len(arr)
    idx = np.arange(1, n + 1)
    denom = n * arr.sum()
    return float((2 * (idx * arr).sum() - (n + 1) * arr.sum()) / denom) if denom else np.nan


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS and c != 'playerName']


def build_ml_row(team, season, player_df, reg_wins, reg_losses) -> dict:
    row = {
        'season':          season,
        'team':            team,
        'reg_season_wins': reg_wins   if reg_wins   >= 0 else np.nan,
        'reg_losses':      reg_losses if reg_losses >= 0 else np.nan,
    }
    if player_df.empty:
        return row

    feature_cols = get_feature_cols(player_df)

    for i in range(1, TOP_N_PLAYERS + 1):
        slot = f'p{i}'
        if i <= len(player_df):
            p = player_df.iloc[i - 1]
            row[f'{slot}_name'] = p.get('playerName', np.nan)
            for col in feature_cols:
                row[f'{slot}_{col}'] = safe_float(p.get(col, np.nan))
        else:
            row[f'{slot}_name'] = np.nan
            for col in feature_cols:
                row[f'{slot}_{col}'] = np.nan

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
#  PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_team_season(espn_team: str, season: int, standings: dict) -> dict:
    bbref_team = espn_to_bbref(espn_team, season)
    print(f"    [{espn_team}/{bbref_team}] ", end='', flush=True)

    tot_df = get_player_totals(espn_team, season)
    adv_df = get_player_advanced(espn_team, season)

    if tot_df.empty and adv_df.empty:
        print("NO DATA")
        reg_wins, reg_losses = lookup_standing(espn_team, season, standings)
        return {
            'season': season, 'team': espn_team,
            'reg_season_wins': reg_wins if reg_wins >= 0 else np.nan,
            'reg_losses': reg_losses if reg_losses >= 0 else np.nan,
        }

    players = merge_totals_and_advanced(tot_df, adv_df)
    players = filter_min_games(players)
    players = sort_and_trim(players)

    reg_wins, reg_losses = lookup_standing(espn_team, season, standings)
    row = build_ml_row(espn_team, season, players, reg_wins, reg_losses)
    print(f"{reg_wins}W / {reg_losses}L  |  {len(players)} players")
    return row


def collect_season(season: int) -> list:
    cache_path = os.path.join(RAW_DIR, f"season_{season}_rows.json")
    if os.path.exists(cache_path):
        print(f"  Season {season}: loading from cache")
        with open(cache_path) as f:
            return json.load(f)

    active_teams = get_active_teams(season)
    print(f"\n{'='*60}")
    print(f"  Season {season-1}-{str(season)[-2:]}  |  {len(active_teams)} teams")
    print(f"{'='*60}")
    print("  Fetching ESPN standings...")
    standings = get_espn_standings(season)
    print(f"  Got standings for {len(standings)} teams")

    rows = []
    for team in active_teams:
        try:
            rows.append(process_team_season(team, season, standings))
        except Exception as e:
            print(f"ERROR — {e}")
            traceback.print_exc()
            rows.append({'season': season, 'team': team, 'error': str(e)})

    with open(cache_path, 'w') as f:
        json.dump(rows, f, default=str)
    return rows


def build_dataset():
    print("\n" + "="*60)
    print("  NBA ML Dataset Builder — Step 1: Stats")
    print(f"  Seasons  : {SEASONS[0]}-{SEASONS[-1]}  ({len(SEASONS)} seasons)")
    print(f"  Top-N    : {TOP_N_PLAYERS} players per row")
    print(f"  Min games: {MIN_GAMES}")
    print("="*60)

    all_rows = []
    for season in SEASONS:
        rows = collect_season(season)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(
            os.path.join(RAW_DIR, "interim_dataset.csv"), index=False
        )

    df = pd.DataFrame(all_rows)

    meta_cols   = ['season', 'team']
    target_cols = ['reg_season_wins', 'reg_losses']
    team_cols   = sorted([c for c in df.columns if c.startswith('team_')])
    player_cols = [c for c in df.columns
                   if any(c.startswith(f'p{i}_') for i in range(1, TOP_N_PLAYERS + 1))]
    player_cols = sorted(player_cols, key=lambda c: (int(c.split('_')[0][1:]), c))
    other_cols  = [c for c in df.columns
                   if c not in meta_cols + target_cols + team_cols + player_cols]

    ordered = meta_cols + target_cols + team_cols + player_cols + other_cols
    df = df[[c for c in ordered if c in df.columns]]

    out_path = os.path.join(FINAL_DIR, "nba_ml_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df)} rows x {len(df.columns)} cols → {out_path}")
    print(f"  Next: run nba_physical_collector.py to append physical attributes")
    return df, out_path


if __name__ == '__main__':
    t0 = datetime.now()
    print(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    build_dataset()
    print(f"Elapsed: {datetime.now() - t0}")