"""
Join Player Physical Attributes into NBA ML Dataset
=====================================================
Adds 7 physical columns per player slot into nba_ml_dataset.csv:
    height_cm, weight_kg, country, college,
    draft_year, draft_round, draft_number

Primary source: nba_data/final/player_dataset.csv
    player_height (cm) -> height_cm
    player_weight (kg) -> weight_kg
    country            -> country
    college            -> college
    draft_year         -> draft_year
    draft_round        -> draft_round
    draft_number       -> draft_number

Fallback source: nba_data/cache/player_physical_cache.json
    Used ONLY for players not found in the CSV.
    Maps into the exact same 7 output columns:
    height_inches (in) -> height_cm  (converted: x 2.54)
    weight_lbs    (lb) -> weight_kg  (converted: x 0.453592)
    country            -> country
    school             -> college
    draft_year         -> draft_year
    draft_round        -> draft_round
    draft_number       -> draft_number

The cache adds NO extra columns — only fills the same 7.
Columns already present in the dataset are never overwritten.

Usage:
    python join_physical.py
"""

import pandas as pd
import numpy as np
import os, re, json, unicodedata, difflib
from datetime import datetime

# ----------------------------------------------------------------
#  CONFIGURATION
# ----------------------------------------------------------------
DATASET_PATH  = os.path.join("nba_data", "final", "nba_ml_dataset.csv")
PHYSICAL_PATH = os.path.join("nba_data", "final", "player_dataset.csv")
CACHE_PATH    = os.path.join("nba_data", "cache", "player_physical_cache.json")

FUZZY_THRESHOLD = 0.85

NAME_OVERRIDES = {
    "Nene":              "Nene Hilario",
    "Harry Giles":       "Harry Giles III",
}

# The 7 output columns this script adds — these are the ONLY columns ever written.
OUTPUT_COLS = ["height_cm", "weight_kg", "country", "college",
               "draft_year", "draft_round", "draft_number"]

# CSV column -> output column  (values already in cm/kg)
PRIMARY_MAP = {
    "player_height": "height_cm",
    "player_weight": "weight_kg",
    "country":       "country",
    "college":       "college",
    "draft_year":    "draft_year",
    "draft_round":   "draft_round",
    "draft_number":  "draft_number",
}

# Cache JSON key -> (output column, conversion function)
# Maps ONLY into the same OUTPUT_COLS — no extras.
def _to_cm(v):
    try:    return round(float(v) * 2.54, 2)
    except: return None

def _to_kg(v):
    try:    return round(float(v) * 0.453592, 2)
    except: return None

def _pass(v):
    return v if v not in (None, "", "Undrafted") else None

def _pass_draft(v):
    # Keep "Undrafted" as-is, pass through everything else
    return v if v not in (None, "") else None

CACHE_MAP = {
    "height_inches": ("height_cm",    _to_cm),
    "weight_lbs":    ("weight_kg",    _to_kg),
    "country":       ("country",      _pass),
    "school":        ("college",      _pass),
    "draft_year":    ("draft_year",   _pass_draft),
    "draft_round":   ("draft_round",  _pass_draft),
    "draft_number":  ("draft_number", _pass_draft),
}


# ================================================================
#  NAME NORMALISATION
# ================================================================

def normalise(name: str) -> str:
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-zA-Z\- ]", "", ascii_str)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


# ================================================================
#  BUILD LOOKUPS
# ================================================================

def build_primary_lookup(phys: pd.DataFrame):
    # Find name column
    name_col = next(
        (c for c in ["normalized_name", "name", "player_name"]
         if c in phys.columns),
        None
    )
    if name_col is None:
        ci = {c.lower(): c for c in phys.columns}
        name_col = ci.get("normalized_name") or ci.get("name")
    if name_col is None:
        raise ValueError(f"No name column found in {PHYSICAL_PATH}. Columns: {list(phys.columns)}")

    available = {src: out for src, out in PRIMARY_MAP.items() if src in phys.columns}
    missing   = [src for src in PRIMARY_MAP if src not in phys.columns]
    if missing:
        print(f"  [WARN] Missing from player_dataset.csv (skipped): {missing}")

    norm_to_row  = {}
    norm_to_orig = {}
    for _, row in phys.iterrows():
        orig = str(row[name_col]).strip()
        key  = normalise(orig)
        if key:
            norm_to_row[key]  = {out: row[src] for src, out in available.items()}
            norm_to_orig[key] = orig

    return norm_to_row, norm_to_orig


def build_cache_lookup(cache: dict):
    norm_to_row  = {}
    norm_to_orig = {}
    for orig, data in cache.items():
        if not isinstance(data, dict) or not data:
            continue
        key = normalise(orig)
        if not key:
            continue
        row = {}
        for json_key, (out_col, fn) in CACHE_MAP.items():
            val = fn(data.get(json_key))
            if val is not None:
                row[out_col] = val
        if row:
            norm_to_row[key]  = row
            norm_to_orig[key] = orig

    return norm_to_row, norm_to_orig


# ================================================================
#  MULTI-STAGE NAME MATCHING
# ================================================================

def find_match(name, norm_to_row, norm_to_orig, keys):
    if not isinstance(name, str) or not name:
        return None, "empty"

    if name in NAME_OVERRIDES:
        k = normalise(NAME_OVERRIDES[name])
        if k in norm_to_row:
            return norm_to_row[k], f"override->{NAME_OVERRIDES[name]}"
        return None, f"override '{NAME_OVERRIDES[name]}' not in source"

    norm = normalise(name)

    if norm in norm_to_row:
        return norm_to_row[norm], "exact"

    parts = norm.split()
    if len(parts) >= 2:
        last, initial = parts[-1], parts[0][0]
        cands = {k: v for k, v in norm_to_row.items()
                 if k.split()[-1] == last and k.split()[0][0] == initial}
        if len(cands) == 1:
            k = next(iter(cands))
            return cands[k], f"last+initial->{norm_to_orig[k]}"
        if len(cands) > 1 and len(parts[0]) >= 2:
            two = parts[0][:2]
            narrow = {k: v for k, v in cands.items() if k.split()[0][:2] == two}
            if len(narrow) == 1:
                k = next(iter(narrow))
                return narrow[k], f"last+2init->{norm_to_orig[k]}"

    hits = difflib.get_close_matches(norm, keys, n=1, cutoff=FUZZY_THRESHOLD)
    if hits:
        k = hits[0]
        return norm_to_row[k], f"fuzzy->{norm_to_orig[k]}"

    closest = difflib.get_close_matches(norm, keys, n=1, cutoff=0.0)
    hint = f"closest: '{norm_to_orig[closest[0]]}'" if closest else "no close match"
    return None, f"not found ({hint})"


# ================================================================
#  SLOT NAME COLUMNS
# ================================================================

def get_name_cols(df: pd.DataFrame) -> list:
    return sorted(
        [c for c in df.columns
         if c.endswith("_name") and c[0] == "p" and c[1:-5].isdigit()],
        key=lambda c: int(c[1:-5])
    )

def extract_unique_names(df: pd.DataFrame) -> list:
    all_names = pd.concat([df[c] for c in get_name_cols(df)], ignore_index=True)
    return sorted(all_names.dropna().unique().tolist())


# ================================================================
#  JOIN: insert the 7 physical cols right after each pN_name
# ================================================================

def join_physical(df: pd.DataFrame, name_to_physical: dict, cols_to_add: list) -> pd.DataFrame:
    df = df.copy()

    for name_col in get_name_cols(df):
        slot = name_col[:-5]
        for phys_col in cols_to_add:
            out_col = f"{slot}_{phys_col}"
            if out_col in df.columns:
                continue   # already present — never overwrite
            df[out_col] = df[name_col].map(
                lambda n, pc=phys_col: (
                    name_to_physical[n].get(pc, np.nan)
                    if isinstance(n, str) and n in name_to_physical
                    else np.nan
                )
            )

    # Re-order: non-player cols | per slot: name -> new physical -> existing stats
    non_player = [c for c in df.columns
                  if not (c[0] == "p" and "_" in c
                          and c.split("_")[0][1:].isdigit())]
    slots_seen = []
    seen = set()
    for c in df.columns:
        if c[0] == "p" and "_" in c:
            prefix = c.split("_")[0]
            if prefix not in seen and prefix[1:].isdigit():
                seen.add(prefix)
                slots_seen.append(prefix)

    final_cols = list(non_player)
    for slot in slots_seen:
        slot_cols = [c for c in df.columns if c.startswith(f"{slot}_")]
        name_c = [c for c in slot_cols if c == f"{slot}_name"]
        phys_c = [c for c in slot_cols if any(c == f"{slot}_{p}" for p in cols_to_add)]
        stat_c = [c for c in slot_cols if c not in name_c and c not in phys_c]
        final_cols.extend(name_c + phys_c + stat_c)

    return df[[c for c in final_cols if c in df.columns]]


# ================================================================
#  MAIN
# ================================================================

def main():
    print("\n" + "="*60)
    print("  Join Player Physical Attributes")
    print(f"  Primary : {PHYSICAL_PATH}")
    print(f"  Fallback: {CACHE_PATH}")
    print(f"  Dataset : {DATASET_PATH}")
    print("="*60)

    for path in [DATASET_PATH, PHYSICAL_PATH]:
        if not os.path.exists(path):
            print(f"\n  ERROR: {path} not found.")
            return

    df   = pd.read_csv(DATASET_PATH)
    phys = pd.read_csv(PHYSICAL_PATH)
    print(f"\n  ML dataset    : {len(df)} rows x {len(df.columns)} cols")
    print(f"  Physical CSV  : {len(phys)} rows x {len(phys.columns)} cols")

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"  Physical cache: {len(cache)} players")
    else:
        print(f"  [WARN] Cache not found at {CACHE_PATH} — fallback disabled")

    # Only add columns not already present
    cols_to_add = [c for c in OUTPUT_COLS if f"p1_{c}" not in df.columns]
    already     = [c for c in OUTPUT_COLS if f"p1_{c}" in df.columns]

    if already:
        print(f"\n  Already in dataset (skipped) : {already}")
    print(f"  Output columns to insert     : {cols_to_add}")

    if not cols_to_add:
        print("\n  Nothing to do — all physical columns already present.")
        return

    # Build lookups
    primary_norm, primary_orig = build_primary_lookup(phys)
    cache_norm,   cache_orig   = build_cache_lookup(cache)
    primary_keys = list(primary_norm.keys())
    cache_keys   = list(cache_norm.keys())
    print(f"\n  Primary index : {len(primary_norm)} players")
    print(f"  Cache index   : {len(cache_norm)} players")

    # Match every unique name
    unique_names = extract_unique_names(df)
    print(f"  Unique names  : {len(unique_names)}\n")

    name_to_physical = {}
    counts    = {"primary": 0, "cache": 0, "not_found": 0}
    unmatched = []

    for name in unique_names:
        row, method = find_match(name, primary_norm, primary_orig, primary_keys)
        if row is not None:
            name_to_physical[name] = row
            counts["primary"] += 1
            continue

        primary_fail = method

        row, method = find_match(name, cache_norm, cache_orig, cache_keys)
        if row is not None:
            name_to_physical[name] = row
            counts["cache"] += 1
            print(f"  [fallback] '{name}' -> cache [{method}]")
            continue

        name_to_physical[name] = {}
        counts["not_found"] += 1
        unmatched.append((name, primary_fail, method))

    print(f"\n  Matched via CSV   : {counts['primary']}")
    print(f"  Matched via cache : {counts['cache']}")
    print(f"  Not found         : {counts['not_found']}")

    if unmatched:
        print(f"\n  {'='*50}")
        print(f"  NOT FOUND IN EITHER SOURCE ({len(unmatched)}):")
        print(f"  {'='*50}")
        for name, pri, cac in unmatched:
            print(f"  '{name}'")
            print(f"      CSV   : {pri}")
            print(f"      cache : {cac}")
        print(f"\n  Add entries to NAME_OVERRIDES at the top to fix.")

    print(f"\n  Inserting columns...")
    df_out = join_physical(df, name_to_physical, cols_to_add)
    df_out.to_csv(DATASET_PATH, index=False)
    print(f"  Saved -> {DATASET_PATH}")
    print(f"  {len(df_out)} rows x {len(df_out.columns)} cols  (+{len(df_out.columns)-len(df.columns)} cols)")

    print(f"\n  Fill rates (p1 slot):")
    for col in cols_to_add:
        fc = f"p1_{col}"
        if fc in df_out.columns:
            n = df_out[fc].notna().sum()
            print(f"    {fc:<30}  {n}/{len(df_out)}  ({n/len(df_out)*100:.0f}%)")

    print("="*60)


if __name__ == "__main__":
    t0 = datetime.now()
    print(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print(f"\nElapsed: {datetime.now() - t0}")