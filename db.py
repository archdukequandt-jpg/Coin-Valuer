from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SEED_FILE = DATA_DIR / "coin_db_seed.csv"
CACHE_FILE = DATA_DIR / "coin_db_cache.csv"

# Canonical columns for the CSV-backed "database".
# We keep legacy columns (metal/metal2/pct2) for compatibility with older UI / valuation logic,
# but the authoritative composition labels for the NN are metal_1..metal_3 + pct_1..pct_3.
DEFAULT_COLUMNS: List[str] = [
    # identity / filters
    "country", "denomination", "currency",
    "year_start", "year_end",
    # physical specs
    "mass_g", "diameter_mm", "thickness_mm", "hole_mm",
    "color_hint",
    # human notes / provenance
    "notes",
    "composition_source",

    # NEW: explicit composition (3-component mixture; pct_* sum to 1.0)
    "metal_1", "pct_1",
    "metal_2", "pct_2",
    "metal_3", "pct_3",

    # Legacy / compatibility fields (used by some UI logic)
    "metal", "metal2", "pct2",
]

@dataclass
class CoinRow:
    country: Optional[str]
    denomination: Optional[str]
    currency: Optional[str]
    year_start: Optional[int]
    year_end: Optional[int]
    mass_g: Optional[float]
    diameter_mm: Optional[float]
    thickness_mm: Optional[float]
    hole_mm: Optional[float]
    color_hint: Optional[str]
    notes: Optional[str]
    composition_source: Optional[str]
    metal_1: Optional[str]
    pct_1: Optional[float]
    metal_2: Optional[str]
    pct_2: Optional[float]
    metal_3: Optional[str]
    pct_3: Optional[float]
    metal: Optional[str]
    metal2: Optional[str]
    pct2: Optional[float]

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    # Type normalization
    for c in ["mass_g","diameter_mm","thickness_mm","hole_mm","pct_1","pct_2","pct_3","pct2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["year_start","year_end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill composition columns with safe defaults (no NA)
    df["metal_1"] = df["metal_1"].fillna("unknown").astype(str)
    df["metal_2"] = df["metal_2"].fillna("none").astype(str)
    df["metal_3"] = df["metal_3"].fillna("none").astype(str)
    df["pct_1"] = df["pct_1"].fillna(1.0)
    df["pct_2"] = df["pct_2"].fillna(0.0)
    df["pct_3"] = df["pct_3"].fillna(0.0)

    # Normalize pct sums to 1.0
    s = (df["pct_1"] + df["pct_2"] + df["pct_3"]).replace(0, pd.NA)
    df["pct_1"] = (df["pct_1"] / s).fillna(1.0)
    df["pct_2"] = (df["pct_2"] / s).fillna(0.0)
    df["pct_3"] = (df["pct_3"] / s).fillna(0.0)

    # Populate legacy fields if missing
    if "metal" in df.columns:
        df["metal"] = df["metal"].fillna(df["metal_1"])
    if "metal2" in df.columns:
        df["metal2"] = df["metal2"].fillna(df["metal_2"])
    if "pct2" in df.columns:
        df["pct2"] = df["pct2"].fillna(df["pct_2"])

    df["composition_source"] = df["composition_source"].fillna("seed").astype(str)
    df["notes"] = df["notes"].fillna("").astype(str)
    df["color_hint"] = df["color_hint"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["denomination"] = df["denomination"].fillna("").astype(str)
    df["currency"] = df["currency"].fillna("").astype(str)

    return df[DEFAULT_COLUMNS].copy()

def load_seed_df() -> pd.DataFrame:
    if SEED_FILE.exists():
        return _ensure_columns(pd.read_csv(SEED_FILE))
    return pd.DataFrame(columns=DEFAULT_COLUMNS)

def load_cache_df() -> pd.DataFrame:
    if CACHE_FILE.exists():
        return _ensure_columns(pd.read_csv(CACHE_FILE))
    return pd.DataFrame(columns=DEFAULT_COLUMNS)

def save_cache_df(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_columns(df).to_csv(CACHE_FILE, index=False)

def load_db_df() -> pd.DataFrame:
    seed = load_seed_df()
    cache = load_cache_df()
    if seed.empty and cache.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    df = pd.concat([seed, cache], ignore_index=True)

    # Dedupe: keep last by key
    df = df.drop_duplicates(subset=["country","denomination","year_start","year_end"], keep="last")
    return _ensure_columns(df)

# Backwards-compatible API expected by app
def current_db() -> pd.DataFrame:
    return load_db_df()

# Placeholder: Wikipedia enrichment (kept for UI compatibility; optional)
def enrich_composition_from_wikipedia(overwrite_existing: bool = False, max_pages: int = 6) -> pd.DataFrame:
    # In this offline environment we keep this as a no-op that returns the cache+seed merged.
    # You can extend this later with requests + parsing if desired.
    df = load_db_df()
    save_cache_df(df)
    return df
