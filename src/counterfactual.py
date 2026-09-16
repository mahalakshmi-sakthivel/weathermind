"""
WeatherMind — actionable counterfactual engine.

This is the module that carries the project's contribution claim, so it is
worth being precise about what it does and does not do.

It does NOT answer "what should this neighbourhood change to be safe?" —
elevation, drainage condition and imperviousness are not things an officer can
alter before tonight's rain. They are IMMUTABLE and are frozen throughout.

It answers two operational questions instead:

1. THRESHOLD BAND — "how much rain does it take before this area crosses into
   HIGH risk?" Answered as an uncertainty band (the range over which the forest
   shifts from mostly-below to mostly-above), not a single number, because a
   single number would imply a precision the model does not have.

2. DIVERSE RECOURSE — "which combinations of forecastable conditions would put
   us in a different risk class?" Answered by searching only the actionable
   features, scoring candidates by a normalised change cost, and discarding any
   that fail the plausibility filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .model import p_at_least, predict_batch
from .plausibility import check as plausibility_check

# Features the search is allowed to move. A strict subset of ACTIONABLE_FEATURES:
# temperature and pressure are forecastable but are not useful levers to reason
# about operationally, so they are held at the forecast value.
SEARCH_FEATURES = ["rainfall_mm", "duration_hr", "wind_speed_kmph", "antecedent_rain_3d_mm"]

# Probability levels defining the uncertainty band around a threshold.
BAND_LEVELS = (0.35, 0.50, 0.65)


@dataclass
class ThresholdBand:
    feature: str
    target_class: int
    lower: float | None
    central: float | None
    upper: float | None
    current_value: float
    current_probability: float
    curve: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def found(self) -> bool:
        return self.central is not None

    def describe(self) -> str:
        label = config.RISK_CLASSES[self.target_class]
        name = config.FEATURE_LABELS[self.feature]
        if not self.found:
            return (
                f"No {name.lower()} value within the searched range flips this area "
                f"into {label}. The transition does not occur under any plausible "
                "value of this variable alone."
            )
        unit = name.split("(")[-1].rstrip(")") if "(" in name else ""
        if self.current_value >= (self.lower or 0):
            return (
                f"Already past the {label} transition: the boundary sits between "
                f"{self.lower:.0f} and {self.upper:.0f} {unit}, and the forecast is "
                f"{self.current_value:.0f} {unit}. Risk would fall back below {label} "
                f"only if the total came in under about {self.lower:.0f} {unit}."
            )
        return (
            f"Transition into {label} occurs between {self.lower:.0f} and "
            f"{self.upper:.0f} {unit} (midpoint {self.central:.0f}), holding everything "
            f"else at the current forecast. Currently {self.current_value:.0f} {unit} — "
            f"about {self.lower - self.current_value:.0f} {unit} of headroom."
        )


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """First x where y crosses `level`, linearly interpolated."""
    above = y >= level
    if not above.any() or above.all():
        return None if not above.any() else float(x[0])
    idx = int(np.argmax(above))
    if idx == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[idx - 1], x[idx], y[idx - 1], y[idx]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def threshold_band(
    bundle,
    record: dict,
    target_class: int = 2,
    feature: str = "rainfall_mm",
    steps: int = 120,
) -> ThresholdBand:
    """Sweep one actionable feature and locate the risk transition band."""
    lo, hi, _ = config.ACTIONABLE_BOUNDS[feature]
    grid = np.linspace(lo, hi, steps)

    candidates = [dict(record, **{feature: float(v)}) for v in grid]
    probs = p_at_least(bundle, candidates, target_class)

    # Smooth the tree-ensemble staircase slightly so the crossing is stable.
    smoothed = pd.Series(probs).rolling(5, center=True, min_periods=1).mean().to_numpy()

    band = {lvl: _crossing(grid, smoothed, lvl) for lvl in BAND_LEVELS}
    current_p = float(p_at_least(bundle, [record], target_class)[0])

    return ThresholdBand(
        feature=feature,
        target_class=target_class,
        lower=band[BAND_LEVELS[0]],
        central=band[BAND_LEVELS[1]],
        upper=band[BAND_LEVELS[2]],
        current_value=float(record[feature]),
        current_probability=current_p,
        curve=pd.DataFrame({feature: grid, "p_at_least": probs, "smoothed": smoothed}),
    )


# --------------------------------------------------------------------------
# Diverse recourse search
# --------------------------------------------------------------------------
def _normalised_cost(candidate: dict, record: dict) -> float:
    """L1 distance in units of each feature's operating range."""
    total = 0.0
    for f in SEARCH_FEATURES:
        lo, hi, _ = config.ACTIONABLE_BOUNDS[f]
        span = (hi - lo) or 1.0
        total += abs(candidate[f] - record[f]) / span
    return total


def _changes(candidate: dict, record: dict, tol: float = 1e-6) -> dict:
    return {
        f: (record[f], candidate[f])
        for f in SEARCH_FEATURES
        if abs(candidate[f] - record[f]) > max(tol, 0.02 * max(abs(record[f]), 1.0))
    }


def diverse_counterfactuals(
    bundle,
    record: dict,
    city: str,
    target_class: int,
    n_samples: int = 4000,
    k: int = 3,
    max_features_changed: int = 2,
    seed: int = 11,
) -> dict:
    """Search actionable space for minimal, plausible ways to reach a class.

    Returns accepted counterfactuals plus any that the model liked but the
    plausibility filter threw out — showing the rejects is part of the point.
    """
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n_samples):
        cand = dict(record)
        n_change = rng.integers(1, max_features_changed + 1)
        for f in rng.choice(SEARCH_FEATURES, size=n_change, replace=False):
            lo, hi, _ = config.ACTIONABLE_BOUNDS[f]
            # Mix local jitter with global draws: local finds near-boundary
            # recourse, global makes sure distant regions are still reachable.
            if rng.random() < 0.6:
                val = record[f] + rng.normal(0, 0.18 * (hi - lo))
            else:
                val = rng.uniform(lo, hi)
            cand[f] = float(np.clip(val, lo, hi))
        samples.append(cand)

    preds = predict_batch(bundle, samples)
    hits = np.where(preds["risk_class"].to_numpy() == target_class)[0]

    scored = []
    for i in hits:
        cand = samples[int(i)]
        changes = _changes(cand, record)
        if not changes:
            continue
        plaus = plausibility_check(city, cand["rainfall_mm"], cand["duration_hr"])
        scored.append(
            {
                "candidate": cand,
                "changes": changes,
                "cost": _normalised_cost(cand, record),
                "confidence": float(preds.iloc[int(i)]["vote_agreement"]),
                "plausibility": plaus,
            }
        )
    scored.sort(key=lambda s: s["cost"])

    accepted, rejected, seen = [], [], set()
    for s in scored:
        key = tuple(sorted(s["changes"].keys()))
        if s["plausibility"].plausible:
            if key in seen or len(accepted) >= k:
                continue
            seen.add(key)
            accepted.append(s)
        elif len(rejected) < 2:
            rejected.append(s)
        if len(accepted) >= k and rejected:
            break

    return {
        "target_class": target_class,
        "target_label": config.RISK_CLASSES[target_class],
        "accepted": accepted,
        "rejected": rejected,
        "n_candidates": n_samples,
        "n_reaching_target": int(len(hits)),
    }


def describe_counterfactual(item: dict) -> str:
    """Render one counterfactual as a sentence an officer can read aloud."""
    parts = []
    for feat, (old, new) in item["changes"].items():
        label = config.FEATURE_LABELS[feat]
        verb = "rises to" if new > old else "falls to"
        parts.append(f"{label} {verb} {new:.0f} (from {old:.0f})")
    return " and ".join(parts)
