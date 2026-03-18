"""Build the lightweight roster/ player-history JSON that the React UI consumes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


INI_DATA = Path("nba_data_2/final/nba_ml_dataset.csv")
OUT_ROSTERS = Path("web/data/rosters.json")
def safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def build() -> Dict[str, Any]:
    df = pd.read_csv(INI_DATA)
    df["team"] = df["team"].str.strip()

    years = sorted({int(season) for season in df["season"].dropna().unique()})
    teams = sorted({team for team in df["team"].dropna().unique()})

    player_histories: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    rosters: list[Dict[str, Any]] = []

    for _, row in df.iterrows():
        season = int(row["season"])
        team = row["team"]
        roster: Dict[str, Any] = {
            "season": season,
            "team": team,
            "regSeasonWins": safe_value(row["reg_season_wins"]),
            "players": [],
        }
        seen_names = set()

        for slot in range(1, 9):
            name_col = f"p{slot}_name"
            name = row.get(name_col, "")
            if isinstance(name, str):
                name = name.strip()
            else:
                name = ""

            prefix = f"p{slot}_"
            stats = {}
            for col in df.columns:
                if col.startswith(prefix):
                    key = col[len(prefix) :]
                    if key == "name":
                        continue
                    stats[key] = safe_value(row.get(col))

            if name in seen_names:
                # Drop duplicate entries within the same roster row
                name = ""
                stats = {}
            else:
                if name:
                    seen_names.add(name)

            roster["players"].append(
                {
                    "slot": slot,
                    "name": name,
                    "stats": stats,
                }
            )

            if not name:
                continue

            player_histories[name].append(
                {
                    "season": season,
                    "team": team,
                    "stats": stats,
                }
            )

        rosters.append(roster)

    for history in player_histories.values():
        history.sort(key=lambda entry: entry["season"])

    return {
        "years": years,
        "teams": teams,
        "rosters": rosters,
        "playerHistories": player_histories,
    }


def main() -> None:
    OUT_ROSTERS.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    with OUT_ROSTERS.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)
    print(f"Saved roster metadata to {OUT_ROSTERS}")


if __name__ == "__main__":
    main()
