"""
Generate the labelled training set for the WeatherMind risk model.

Why this exists
---------------
There is no public dataset that labels *neighbourhood-level flood risk* for
Indian cities. The Kaggle rainfall datasets give you weather, not consequences.
So the prototype samples realistic weather scenarios for each curated area and
labels them with an explicit, readable hydrological score (see
`src/features.latent_risk_score`).

That makes the pipeline honest: the Random Forest is a real model trained on
real feature interactions, and the assumptions baked into the labels are a
50-line function anyone can read and argue with — rather than hidden inside a
model nobody can inspect.

Replace this with observed incident labels (or a real Kaggle export at
`data/kaggle_rainfall.csv`) whenever you have them; nothing else changes.

Usage:
    python scripts/generate_training_data.py [--rows-per-area 1200] [--seed 7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import load_baselines, load_spatial  # noqa: E402
from src.features import latent_risk_score, score_to_class  # noqa: E402


def sample_weather(n: int, wetness: float, rng: np.random.Generator) -> pd.DataFrame:
    """Sample a plausible spread of event scenarios for one city.

    `wetness` scales rainfall magnitude by city (Chennai is wetter than the
    inland districts). The spread deliberately covers everything from dry days
    to extreme events so the decision boundary is learned across the full range
    the what-if simulator can reach.
    """
    duration = np.clip(rng.lognormal(mean=2.2, sigma=0.7, size=n), 1, 48)

    # Rainfall is sampled for COVERAGE of the decision region, not to match
    # climatology: a realistic day-by-day distribution is ~90% light rain,
    # which starves the interesting part of the decision boundary. Real-world
    # likelihood is enforced separately and explicitly by the plausibility
    # filter (src/plausibility.py), which is the right place for it.
    u = rng.random(n)
    rainfall = np.where(
        u < 0.32, rng.uniform(0, 18, n),                 # light / dry spells
        np.where(u < 0.76, rng.uniform(18, 90, n),       # moderate to heavy
                 rng.uniform(90, 340, n)),               # very heavy to extreme
    )
    rainfall = np.clip(rainfall * (0.75 + 0.5 * wetness), 0, 400)

    return pd.DataFrame(
        {
            "rainfall_mm": np.round(rainfall, 1),
            "duration_hr": np.round(duration, 1),
            "wind_speed_kmph": np.round(np.clip(rng.normal(22, 12, n), 0, 120), 1),
            "humidity_pct": np.round(
                np.clip(62 + 0.09 * rainfall + rng.normal(0, 8, n), 20, 100), 1
            ),
            "temperature_c": np.round(
                np.clip(30 - 0.025 * rainfall + rng.normal(0, 3, n), 15, 45), 1
            ),
            "pressure_hpa": np.round(
                np.clip(1010 - 0.02 * rainfall + rng.normal(0, 3, n), 985, 1020), 1
            ),
            "antecedent_rain_3d_mm": np.round(
                np.clip(rng.lognormal(2.4, 1.0, n) * wetness, 0, 300), 1
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-area", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    spatial = load_spatial()
    baselines = load_baselines().set_index("city")

    # Normalise city wetness around Coimbatore.
    ref = float(baselines["annual_normal_mm"].median())

    frames = []
    for _, prof in spatial.iterrows():
        wetness = float(baselines.loc[prof["city"], "annual_normal_mm"]) / ref
        weather = sample_weather(args.rows_per_area, wetness, rng)

        for feat in config.IMMUTABLE_FEATURES:
            weather[feat] = float(prof[feat])
        weather["city"] = prof["city"]
        weather["area"] = prof["area"]

        score = latent_risk_score(weather)
        # Measurement / model noise so the classes overlap near the boundary —
        # this is what gives the confidence score something real to express.
        noisy = score * (1 + rng.normal(0, 0.12, len(score))) + rng.normal(0, 1.5, len(score))
        weather["latent_score"] = np.round(score, 3)
        weather["risk_class"] = score_to_class(noisy)
        frames.append(weather)

    data = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=args.seed)
    data.to_csv(config.TRAINING_CSV, index=False)

    counts = data["risk_class"].value_counts().sort_index()
    print(f"Wrote {len(data):,} rows -> {config.TRAINING_CSV}")
    for cls, label in config.RISK_CLASSES.items():
        n = int(counts.get(cls, 0))
        print(f"  {label:<9} {n:>6,}  ({n / len(data):.1%})")


if __name__ == "__main__":
    main()
