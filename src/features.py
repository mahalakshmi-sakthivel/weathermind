"""
WeatherMind — feature engineering.

Two jobs:

1. `engineer_features` turns a raw record (weather + spatial) into the exact
   column order the Random Forest expects. Used identically at training time,
   prediction time, counterfactual-search time and risk-surface time, so there
   is no train/serve skew.

2. `latent_risk_score` is a transparent, physically-motivated hydrological
   score. It is used ONLY to label the generated training set (see
   `scripts/generate_training_data.py`). The deployed model never calls it —
   the Random Forest has to learn the relationship from data. Keeping the
   label-generating function explicit and readable is deliberate: reviewers can
   see exactly what assumptions the synthetic ground truth encodes, and it is
   trivial to swap the whole thing out for observed incident labels later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_ORDER

# Class cut-points on the latent score. Fixed (not per-area quantiles) so the
# risk-response surface means the same thing in every city.
RISK_CUTS = (15.0, 65.0)


def _as_frame(record) -> pd.DataFrame:
    if isinstance(record, pd.DataFrame):
        return record.copy()
    if isinstance(record, pd.Series):
        return record.to_frame().T.copy()
    if isinstance(record, (list, tuple)):
        return pd.DataFrame(list(record))
    return pd.DataFrame([dict(record)])


def drainage_capacity(df: pd.DataFrame) -> pd.Series:
    """Approximate how many mm/hr the area can shed before water stands.

    Driven by drainage vulnerability (network condition + encroachment),
    slope (gravity does some of the work) and soil infiltration.
    """
    return (
        2.0
        + 14.0 * (1.0 - df["drainage_vulnerability"])
        + 2.5 * df["slope_pct"]
        + 6.0 * df["soil_infiltration"]
    )


def engineer_features(record) -> pd.DataFrame:
    """Add derived features and return columns in `FEATURE_ORDER`."""
    df = _as_frame(record)

    duration = df["duration_hr"].clip(lower=0.5)
    df["intensity_mm_per_hr"] = df["rainfall_mm"] / duration

    antecedent_factor = 1.0 + 0.35 * (df["antecedent_rain_3d_mm"] / 150.0).clip(upper=1.0)
    df["effective_runoff_mm"] = (
        df["rainfall_mm"]
        * (0.25 + 0.75 * df["imperviousness"])
        * (1.0 - 0.45 * df["soil_infiltration"])
        * antecedent_factor
    )

    capacity = drainage_capacity(df)
    df["drainage_deficit"] = df["intensity_mm_per_hr"] / capacity

    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    return df[FEATURE_ORDER]


def latent_risk_score(df: pd.DataFrame) -> pd.Series:
    """Transparent hydrological score used to label the generated dataset."""
    feats = engineer_features(df)

    ratio = feats["drainage_deficit"]  # intensity / capacity
    low_lying = ((60.0 - feats["elevation_m"]) / 60.0).clip(lower=0.0, upper=1.0)

    ponding_term = 50.0 * np.power(ratio.clip(lower=0.0), 1.4)
    volume_term = (
        0.30
        * feats["effective_runoff_mm"]
        * (1.0 + 0.8 * low_lying)
        / (1.0 + 2.0 * feats["slope_pct"])
    )
    wind_term = 0.03 * feats["wind_speed_kmph"]

    # Capped: beyond this everything is unambiguously HIGH, and an unbounded
    # tail would make the noise model behave badly during label generation.
    return (ponding_term + volume_term + wind_term).clip(upper=250.0)


def score_to_class(score) -> np.ndarray:
    """Map the latent score onto LOW (0) / MODERATE (1) / HIGH (2)."""
    lo, hi = RISK_CUTS
    score = np.asarray(score, dtype=float)
    return np.where(score < lo, 0, np.where(score < hi, 1, 2)).astype(int)
