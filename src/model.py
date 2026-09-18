"""
WeatherMind — risk model.

A Random Forest classifier over the combined weather + spatial feature vector,
returning a risk class plus two different notions of confidence:

* `probability`      — averaged leaf distributions (`predict_proba`)
* `vote_agreement`   — the fraction of trees that individually vote for the
                       winning class

They disagree in exactly the interesting cases: when the forest is split near a
threshold, vote agreement drops faster than probability does. WeatherMind
reports vote agreement as the headline confidence because an officer reading
"9 of 10 models agree" is reading something more honest than a smoothed
probability.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from . import config
from .data_loader import load_training_data
from .features import engineer_features

try:  # pragma: no cover
    import streamlit as st

    def _cache_resource(func):
        return st.cache_resource(show_spinner="Training risk model…")(func)

except Exception:  # pragma: no cover

    def _cache_resource(func):
        return functools.lru_cache(maxsize=1)(func)


@dataclass
class ModelBundle:
    model: RandomForestClassifier
    features: list
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_model(
    data: Optional[pd.DataFrame] = None,
    n_estimators: int = 300,
    max_depth: Optional[int] = 14,
    min_samples_leaf: int = 4,
    random_state: int = 42,
) -> ModelBundle:
    """Fit the Random Forest and report held-out metrics."""
    data = load_training_data() if data is None else data
    X = engineer_features(data)
    y = data["risk_class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "report": classification_report(
            y_test, y_pred,
            target_names=[config.RISK_CLASSES[c] for c in sorted(y.unique())],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": dict(
            zip(X.columns, np.round(rf.feature_importances_, 4).tolist())
        ),
    }

    return ModelBundle(
        model=rf,
        features=list(X.columns),
        metrics=metrics,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_model(bundle: ModelBundle) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)


@_cache_resource
def load_model() -> ModelBundle:
    """Load the persisted model, training and caching one if absent.

    This is what makes a cold deploy on Streamlit Community Cloud work with no
    build step: the first request trains in a few seconds, then it is cached.
    """
    if config.MODEL_PATH.exists():
        try:
            return joblib.load(config.MODEL_PATH)
        except Exception:
            pass  # version mismatch on a fresh host — just retrain
    bundle = train_model()
    try:
        save_model(bundle)
    except OSError:
        pass  # read-only filesystem is fine, the cache holds it
    return bundle


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def get_tree_vote_confidence(rf_model, X_instance) -> float:
    """
    Returns the fraction of individual trees in the Random Forest
    that voted for the ensemble's final predicted class.
    This is what's displayed as the 'confidence score' in the UI.
    """
    model = rf_model.model if hasattr(rf_model, "model") else rf_model
    tree_predictions = np.array([
        tree.predict(X_instance)[0] for tree in model.estimators_
    ])
    final_prediction = model.predict(X_instance)[0]
    agreement = float(np.mean(tree_predictions == final_prediction))
    return agreement  # e.g., 0.87 → displayed as "87% confidence"


def _tree_vote_matrix(rf: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """(n_trees, n_samples) matrix of per-tree class votes."""
    idx = np.array([est.predict(X.values) for est in rf.estimators_], dtype=int)
    return rf.classes_.take(idx)


def predict_batch(bundle: ModelBundle, records) -> pd.DataFrame:
    """Vectorised prediction. Returns one row per input record."""
    X = engineer_features(records)[bundle.features]
    proba = bundle.model.predict_proba(X)
    votes = _tree_vote_matrix(bundle.model, X)

    n_trees = votes.shape[0]
    classes = bundle.model.classes_
    vote_share = np.stack(
        [(votes == c).sum(axis=0) / n_trees for c in classes], axis=1
    )

    pred_idx = proba.argmax(axis=1)
    pred_class = classes[pred_idx]
    out = pd.DataFrame(
        {
            "risk_class": pred_class,
            "risk_label": [config.RISK_CLASSES[int(c)] for c in pred_class],
            "probability": proba[np.arange(len(proba)), pred_idx],
            "vote_agreement": vote_share[np.arange(len(vote_share)), pred_idx],
        }
    )
    for i, c in enumerate(classes):
        out[f"p_{config.RISK_CLASSES[int(c)].lower()}"] = proba[:, i]
    return out


def predict(bundle: ModelBundle, record: dict) -> dict:
    """Single-record prediction with a confidence narrative."""
    row = predict_batch(bundle, [record]).iloc[0]
    result = row.to_dict()
    result["risk_class"] = int(result["risk_class"])
    result["confidence"] = float(result["vote_agreement"])
    result["confidence_note"] = _confidence_note(result["confidence"])
    result["recommended_posture"] = config.RISK_ACTIONS[result["risk_class"]]
    return result


def _confidence_note(agreement: float) -> str:
    trees = round(agreement * 100)
    if agreement >= 0.85:
        return f"Strong agreement — {trees}% of trees vote this class. Well inside a stable region."
    if agreement >= 0.65:
        return f"Moderate agreement — {trees}% of trees vote this class. Some sensitivity to input error."
    return (
        f"Weak agreement — only {trees}% of trees vote this class. "
        "The scenario sits close to a decision boundary; treat the class as provisional "
        "and read the threshold range below rather than the label alone."
    )


def p_at_least(bundle: ModelBundle, records, target_class: int) -> np.ndarray:
    """P(risk >= target_class) for each record — the risk-surface quantity."""
    X = engineer_features(records)[bundle.features]
    proba = bundle.model.predict_proba(X)
    cols = [i for i, c in enumerate(bundle.model.classes_) if int(c) >= target_class]
    return proba[:, cols].sum(axis=1)
