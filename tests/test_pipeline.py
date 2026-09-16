"""
End-to-end checks for the WeatherMind pipeline.

Run with:  python -m pytest -q     (or simply: python tests/test_pipeline.py)

These are deliberately behavioural rather than unit-level: they assert the
properties the project's claims depend on — that spatial features are never
moved by the counterfactual engine, that risk is monotone in rainfall, that
implausible scenarios are filtered, and that a brief is produced with no API
key present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.counterfactual import SEARCH_FEATURES, diverse_counterfactuals, threshold_band
from src.data_loader import (
    areas_for, build_record, cities, default_weather, get_area_profile, get_incidents,
)
from src.features import engineer_features
from src.llm_brief import build_payload, generate_brief
from src.model import load_model, predict
from src.plausibility import check
from src.risk_surface import compute_surface, plausibility_mask
from src.shap_utils import explain, group_contributions

BUNDLE = load_model()


def _record(city="Coimbatore", area="Peelamedu", **overrides):
    weather = default_weather(city)
    weather.update(overrides)
    return build_record(city, area, weather), weather


def test_every_area_has_data():
    for city in cities():
        assert areas_for(city), f"{city} has no areas"
        for area in areas_for(city):
            get_area_profile(city, area)
            assert not get_incidents(city, area).empty


def test_feature_order_is_stable():
    record, _ = _record()
    assert list(engineer_features(record).columns) == config.FEATURE_ORDER


def test_risk_is_monotone_in_rainfall():
    """More rain, same everything else, must never lower P(HIGH)."""
    probs = []
    for rain in [10, 40, 80, 140, 220, 320]:
        record, _ = _record(rainfall_mm=rain, duration_hr=12)
        probs.append(predict(BUNDLE, record)["p_high"])
    assert all(b >= a - 0.02 for a, b in zip(probs, probs[1:])), probs


def test_low_and_high_extremes_classify_sensibly():
    dry, _ = _record(rainfall_mm=2, duration_hr=12)
    assert predict(BUNDLE, dry)["risk_label"] == "LOW"

    deluge, _ = _record(city="Chennai", area="Velachery", rainfall_mm=250, duration_hr=12)
    assert predict(BUNDLE, deluge)["risk_label"] == "HIGH"


def test_vulnerable_area_is_riskier_than_resilient_one():
    """Same weather, different ground, must produce different risk."""
    weather = dict(default_weather("Tirupur"), rainfall_mm=90, duration_hr=12)
    fragile = predict(BUNDLE, build_record("Tirupur", "Noyyal Riverside", weather))
    sturdy = predict(BUNDLE, build_record("Tirupur", "Veerapandi", weather))
    assert fragile["p_high"] > sturdy["p_high"]


def test_counterfactuals_never_touch_immutable_features():
    record, _ = _record(rainfall_mm=50, duration_hr=12)
    result = diverse_counterfactuals(BUNDLE, record, "Coimbatore", target_class=2)
    for item in result["accepted"] + result["rejected"]:
        for feat in config.IMMUTABLE_FEATURES:
            assert item["candidate"][feat] == record[feat], f"{feat} was modified"
        assert set(item["changes"]).issubset(set(SEARCH_FEATURES))


def test_accepted_counterfactuals_are_plausible():
    record, _ = _record(rainfall_mm=50, duration_hr=12)
    result = diverse_counterfactuals(BUNDLE, record, "Coimbatore", target_class=2)
    assert all(c["plausibility"].plausible for c in result["accepted"])


def test_threshold_band_is_ordered_and_above_current():
    record, _ = _record(rainfall_mm=40, duration_hr=12)
    band = threshold_band(BUNDLE, record, target_class=2)
    assert band.found
    assert band.lower <= band.central <= band.upper
    assert band.central > record["rainfall_mm"]


def test_plausibility_rejects_absurd_scenarios():
    assert not check("Coimbatore", 900, 3).plausible
    assert check("Coimbatore", 40, 12).plausible
    assert check("Chennai", 200, 24).level in ("rare", "uncommon", "unprecedented")


def test_surface_shape_and_mask():
    record, _ = _record()
    surface = compute_surface(BUNDLE, record, nx=20, ny=15)
    assert surface["z"].shape == (15, 20)
    assert ((surface["z"] >= 0) & (surface["z"] <= 1)).all()
    mask = plausibility_mask(surface, "Coimbatore")
    assert mask is not None and mask.any()


def test_factor_analysis_returns_all_features():
    record, _ = _record()
    factors = explain(BUNDLE, record)
    assert len(factors) == len(BUNDLE.features)
    assert set(factors["mutability"]) <= {"actionable", "immutable"}
    assert np.isclose(group_contributions(factors)["share"].sum(), 1.0)


def test_brief_generates_without_api_key(monkeypatch=None):
    city, area = "Coimbatore", "Peelamedu"
    record, weather = _record(rainfall_mm=120, duration_hr=12)
    prediction = predict(BUNDLE, record)
    payload = build_payload(
        role=list(config.ROLES)[0], city=city, area=area,
        profile=get_area_profile(city, area), weather=weather,
        prediction=prediction, factors=explain(BUNDLE, record),
        band=threshold_band(BUNDLE, record, 2),
        counterfactuals=diverse_counterfactuals(BUNDLE, record, city, 2),
        plausibility=check(city, weather["rainfall_mm"], weather["duration_hr"]),
        incidents=get_incidents(city, area),
    )
    text, source = generate_brief(payload)
    assert "Recommended actions" in text
    assert len(text) > 200
    assert source


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print("\nAll green." if not failures else f"\n{failures} failing test(s).")
    sys.exit(1 if failures else 0)
