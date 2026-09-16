"""
WeatherMind — officer feedback loop.

Every assessment an officer confirms or disputes is appended to a CSV with the
full input vector attached. That file is the seed of the recalibration loop: a
growing set of real, locally-observed labels that can eventually replace the
generated training labels entirely.

The retraining itself is intentionally out of scope for the prototype — what is
built here is the capture mechanism and the schema, which is the part that has
to be right from day one.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone

import pandas as pd

from . import config

FIELDS = [
    "timestamp", "role", "city", "area",
    "rainfall_mm", "duration_hr", "wind_speed_kmph", "humidity_pct",
    "temperature_c", "pressure_hpa", "antecedent_rain_3d_mm",
    "predicted_class", "confidence", "officer_verdict", "observed_outcome", "notes",
]


def log_feedback(
    *, role: str, city: str, area: str, weather: dict, prediction: dict,
    verdict: str, observed_outcome: str = "", notes: str = "",
) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not config.FEEDBACK_LOG.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": role, "city": city, "area": area,
        **{f: round(float(weather.get(f, float("nan"))), 2)
           for f in config.ACTIONABLE_FEATURES},
        "predicted_class": prediction["risk_label"],
        "confidence": round(float(prediction["confidence"]), 3),
        "officer_verdict": verdict,
        "observed_outcome": observed_outcome,
        "notes": notes,
    }

    with config.FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def load_feedback() -> pd.DataFrame:
    if not config.FEEDBACK_LOG.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(config.FEEDBACK_LOG)


def feedback_summary() -> dict:
    df = load_feedback()
    if df.empty:
        return {"n": 0, "agreement_rate": None}
    agree = (df["officer_verdict"] == "Accurate").mean()
    return {
        "n": int(len(df)),
        "agreement_rate": float(agree),
        "by_area": df.groupby("area")["officer_verdict"].value_counts().to_dict(),
    }
