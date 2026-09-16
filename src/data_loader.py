"""
WeatherMind — data access layer.

All CSV reading happens here so the rest of the app never touches a path.
Loaders are cached: under Streamlit they use `st.cache_data`, and outside
Streamlit (tests, training scripts) they fall back to `functools.lru_cache`.
"""

from __future__ import annotations

import functools
from typing import Optional

import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Cache shim — works with or without Streamlit installed
# --------------------------------------------------------------------------
try:  # pragma: no cover - depends on runtime
    import streamlit as st

    def _cache(func):
        return st.cache_data(show_spinner=False)(func)

except Exception:  # pragma: no cover

    def _cache(func):
        return functools.lru_cache(maxsize=None)(func)


@_cache
def load_spatial() -> pd.DataFrame:
    """Hand-curated neighbourhood vulnerability table."""
    return pd.read_csv(config.SPATIAL_CSV)


@_cache
def load_incidents() -> pd.DataFrame:
    """Historical incident records used to ground the LLM brief."""
    df = pd.read_csv(config.INCIDENTS_CSV)
    return df.sort_values("event_date", ascending=False)


@_cache
def load_baselines() -> pd.DataFrame:
    """Per-city rainfall statistics used by the plausibility filter."""
    return pd.read_csv(config.BASELINES_CSV)


@_cache
def load_training_data() -> pd.DataFrame:
    """Training set.

    Prefers `data/kaggle_rainfall.csv` if you have dropped a real export in
    (see README), otherwise uses the generated `data/training_data.csv`.
    """
    if config.KAGGLE_CSV.exists():
        return pd.read_csv(config.KAGGLE_CSV)
    if not config.TRAINING_CSV.exists():
        raise FileNotFoundError(
            "No training data found. Run: python scripts/generate_training_data.py"
        )
    return pd.read_csv(config.TRAINING_CSV)


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------
def get_area_profile(city: str, area: str) -> pd.Series:
    """Immutable spatial context for one area."""
    spatial = load_spatial()
    match = spatial[(spatial["city"] == city) & (spatial["area"] == area)]
    if match.empty:
        raise ValueError(f"No spatial record for {area}, {city}")
    return match.iloc[0]


def get_incidents(city: str, area: str, limit: int = 3) -> pd.DataFrame:
    """Most recent historical incidents for an area (falls back to city)."""
    inc = load_incidents()
    rows = inc[(inc["city"] == city) & (inc["area"] == area)]
    if rows.empty:
        rows = inc[inc["city"] == city]
    return rows.head(limit)


def get_baseline(city: str) -> pd.Series:
    base = load_baselines()
    match = base[base["city"] == city]
    if match.empty:
        raise ValueError(f"No rainfall baseline for {city}")
    return match.iloc[0]


def cities() -> list[str]:
    return sorted(load_spatial()["city"].unique().tolist())


def areas_for(city: str) -> list[str]:
    spatial = load_spatial()
    return spatial.loc[spatial["city"] == city, "area"].tolist()


# --------------------------------------------------------------------------
# Record assembly
# --------------------------------------------------------------------------
def build_record(city: str, area: str, weather: dict,
                 profile: Optional[pd.Series] = None) -> dict:
    """Merge actionable weather inputs with the frozen spatial layer."""
    prof = profile if profile is not None else get_area_profile(city, area)
    record = {f: float(weather[f]) for f in config.ACTIONABLE_FEATURES}
    for f in config.IMMUTABLE_FEATURES:
        record[f] = float(prof[f])
    return record


def default_weather(city: str) -> dict:
    """A sensible starting scenario — a moderate northeast-monsoon spell."""
    base = get_baseline(city)
    return {
        "rainfall_mm": float(base["daily_p90_mm"]),
        "duration_hr": 12.0,
        "wind_speed_kmph": 25.0,
        "humidity_pct": 85.0,
        "temperature_c": 26.0,
        "pressure_hpa": 1004.0,
        "antecedent_rain_3d_mm": 30.0,
    }
