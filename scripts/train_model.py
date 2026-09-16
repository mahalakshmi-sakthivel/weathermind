"""
Train and persist the WeatherMind risk model.

Optional — the app trains on first run if no artifact exists. Run this to see
held-out metrics, or to bake a model into the image before deploying.

Usage:
    python scripts/train_model.py [--trees 300] [--depth 14]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.model import save_model, train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--depth", type=int, default=14)
    args = parser.parse_args()

    bundle = train_model(n_estimators=args.trees, max_depth=args.depth)
    save_model(bundle)

    m = bundle.metrics
    print(f"Saved -> {config.MODEL_PATH}")
    print(f"Held-out accuracy: {m['accuracy']:.3f} on {m['n_test']:,} rows")
    print("\nPer-class f1:")
    for label in config.RISK_CLASSES.values():
        row = m["report"].get(label)
        if row:
            print(f"  {label:<9} precision {row['precision']:.3f}  "
                  f"recall {row['recall']:.3f}  f1 {row['f1-score']:.3f}")
    print("\nTop features:")
    top = sorted(m["feature_importance"].items(), key=lambda kv: -kv[1])[:6]
    print(json.dumps(dict(top), indent=2))


if __name__ == "__main__":
    main()
