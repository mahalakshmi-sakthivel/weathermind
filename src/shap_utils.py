"""
WeatherMind — factor analysis ("why this risk class?").

Primary path: SHAP TreeExplainer, which gives exact per-feature contributions
for tree ensembles.

Fallback path: if `shap` is not installed (it is a heavy dependency and
occasionally awkward on free hosting tiers), a local sensitivity analysis is
used instead — each feature is moved to its dataset median one at a time and
the change in the predicted class probability is recorded. Less principled than
SHAP, but per-instance, signed, and never crashes a live demo.

The UI states which path produced the numbers.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_training_data
from .features import engineer_features

try:
    import shap  # type: ignore

    _HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    _HAS_SHAP = False


def explainer_kind() -> str:
    return "SHAP (TreeExplainer)" if _HAS_SHAP else "Local sensitivity (SHAP unavailable)"


# Groups used for the plain-language summary and the LLM prompt.
FEATURE_GROUPS = {
    "Rain magnitude & intensity": [
        "rainfall_mm", "duration_hr", "intensity_mm_per_hr", "effective_runoff_mm",
    ],
    "Drainage & terrain (immutable)": [
        "elevation_m", "slope_pct", "drainage_vulnerability", "imperviousness",
        "soil_infiltration", "drainage_deficit",
    ],
    "Antecedent conditions": ["antecedent_rain_3d_mm"],
    "Other weather": ["wind_speed_kmph", "humidity_pct", "temperature_c", "pressure_hpa"],
}


@functools.lru_cache(maxsize=1)
def _medians() -> pd.Series:
    return engineer_features(load_training_data()).median()


def _expected_level(bundle, X: pd.DataFrame) -> np.ndarray:
    """E[risk class] = 0*P(LOW) + 1*P(MODERATE) + 2*P(HIGH).

    Attribution targets this single scalar rather than the probability of the
    winning class. That matters for MODERATE predictions: a factor pushing the
    scenario toward HIGH *reduces* P(MODERATE), so class-probability
    attribution would label it "lowers risk", which is exactly backwards.
    """
    proba = bundle.model.predict_proba(X)
    weights = np.asarray(bundle.model.classes_, dtype=float)
    return proba @ weights


def _shap_contributions(bundle, X: pd.DataFrame) -> np.ndarray:
    """Per-feature SHAP values, combined into expected-risk-level units."""
    explainer = shap.TreeExplainer(bundle.model)
    values = explainer.shap_values(X, check_additivity=False)
    classes = np.asarray(bundle.model.classes_, dtype=float)

    if isinstance(values, list):                       # older API: list per class
        stacked = np.stack([np.asarray(v)[0] for v in values], axis=-1)
    else:
        values = np.asarray(values)
        if values.ndim == 3:                           # (n, features, classes)
            stacked = values[0]
        else:                                          # binary / single output
            return np.asarray(values[0], dtype=float)
    return stacked @ classes


def _sensitivity_contributions(bundle, X: pd.DataFrame) -> np.ndarray:
    """Change in expected risk level if this feature were typical instead."""
    med = _medians()
    base = _expected_level(bundle, X)[0]

    perturbed = pd.concat([X] * len(X.columns), ignore_index=True)
    for i, col in enumerate(X.columns):
        perturbed.loc[i, col] = med[col]
    return base - _expected_level(bundle, perturbed)


def explain(bundle, record: dict, top_n: int | None = None) -> pd.DataFrame:
    """Per-feature contribution to the expected risk level, largest first."""
    X = engineer_features(record)[bundle.features]

    if _HAS_SHAP:
        contrib = _shap_contributions(bundle, X)
    else:
        contrib = _sensitivity_contributions(bundle, X)

    df = pd.DataFrame(
        {
            "feature": bundle.features,
            "label": [config.FEATURE_LABELS.get(f, f) for f in bundle.features],
            "value": X.iloc[0].values,
            "contribution": contrib,
        }
    )
    df["mutability"] = np.where(
        df["feature"].isin(config.IMMUTABLE_FEATURES), "immutable", "actionable"
    )
    df["direction"] = np.where(df["contribution"] >= 0, "raises risk", "lowers risk")
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    return df.head(top_n) if top_n else df


def group_contributions(factors: pd.DataFrame) -> pd.DataFrame:
    """Roll per-feature contributions up into four readable groups."""
    rows = []
    for group, members in FEATURE_GROUPS.items():
        subset = factors[factors["feature"].isin(members)]
        rows.append({"group": group, "contribution": float(subset["contribution"].sum())})
    out = pd.DataFrame(rows)
    total = out["contribution"].abs().sum() or 1.0
    out["share"] = out["contribution"].abs() / total
    return out.sort_values("contribution", ascending=False).reset_index(drop=True)


def narrate(factors: pd.DataFrame, top_n: int = 3) -> str:
    """One-sentence plain-language summary of the dominant drivers."""
    top = factors.head(top_n)
    parts = [
        f"{r.label} ({r.value:.1f}) {r.direction}"
        for r in top.itertuples()
    ]
    return "Main drivers: " + "; ".join(parts) + "."
