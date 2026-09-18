"""
WeatherMind — hyper-local flood-risk decision support.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.counterfactual import (
    SEARCH_FEATURES,
    describe_counterfactual,
    diverse_counterfactuals,
    threshold_band,
)
from src.data_loader import (
    areas_for,
    build_record,
    cities,
    default_weather,
    get_area_profile,
    get_baseline,
    get_incidents,
)
from src.feedback import feedback_summary, log_feedback
from src.llm_brief import build_payload, generate_brief, generate_decision_brief
from src.model import load_model, predict
from src.plausibility import check as plausibility_check
from src.plausibility import rejected_example
from src.risk_surface import (
    compute_surface,
    factor_figure,
    surface_figure,
    surface_summary,
    threshold_curve_figure,
)
from src.shap_utils import explain, explainer_kind, group_contributions, narrate

st.set_page_config(
    page_title="WeatherMind — City Impact Analysis",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.risk-banner {padding: 1.1rem 1.4rem; border-radius: 12px; color: #fff; margin: .3rem 0 1rem 0;}
.risk-banner h2 {margin: 0; font-size: 1.9rem; letter-spacing: .5px;}
.risk-banner p {margin: .35rem 0 0 0; opacity: .93; font-size: .95rem;}
.chip {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.75rem;
       background:#eef2f7; color:#33415c; margin-right:.3rem;}
.chip-locked {background:#efeaf7; color:#4b3c78;}
.small-note {color:#6b7280; font-size:.82rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached compute helpers
# --------------------------------------------------------------------------
def _key(record: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 4)) for k, v in record.items()))


@st.cache_data(show_spinner=False)
def cached_prediction(record_key: tuple) -> dict:
    return predict(load_model(), dict(record_key))


@st.cache_data(show_spinner=False)
def cached_factors(record_key: tuple) -> pd.DataFrame:
    return explain(load_model(), dict(record_key))


@st.cache_data(show_spinner=False)
def cached_band(record_key: tuple, target_class: int, feature: str):
    return threshold_band(load_model(), dict(record_key), target_class, feature)


@st.cache_data(show_spinner=False)
def cached_surface(record_key: tuple, x_feature: str, y_feature: str, target_class: int):
    return compute_surface(
        load_model(), dict(record_key),
        x_feature=x_feature, y_feature=y_feature, target_class=target_class,
    )


@st.cache_data(show_spinner=False)
def cached_counterfactuals(record_key: tuple, city: str, target_class: int):
    return diverse_counterfactuals(load_model(), dict(record_key), city, target_class)


# --------------------------------------------------------------------------
# Sidebar — role, location, forecast inputs
# --------------------------------------------------------------------------
def reset_scenario() -> None:
    """Drop widget state so the new city's defaults take effect."""
    for feat in config.ACTIONABLE_FEATURES:
        st.session_state.pop(f"w_{feat}", None)
    st.session_state.pop("baseline", None)
    st.session_state.pop("brief", None)


with st.sidebar:
    st.markdown("## 🌧️ WeatherMind")
    st.caption("Hyper-local flood-risk decision support")

    role = st.selectbox("Your role", list(config.ROLES), key="role")
    city = st.selectbox("City", cities(), key="city", on_change=reset_scenario)
    area = st.selectbox("Area", areas_for(city), key="area", on_change=reset_scenario)

    st.divider()
    st.markdown("### Forecast inputs")
    st.caption("Actionable variables — the only things the counterfactual engine may move.")

    defaults = default_weather(city)
    for feat in config.ACTIONABLE_FEATURES:
        st.session_state.setdefault(f"w_{feat}", float(defaults[feat]))

    weather: dict = {}
    for feat in config.ACTIONABLE_FEATURES:
        lo, hi, step = config.ACTIONABLE_BOUNDS[feat]
        weather[feat] = st.slider(
            config.FEATURE_LABELS[feat],
            min_value=float(lo), max_value=float(hi),
            step=float(step), key=f"w_{feat}",
        )

    if "baseline" not in st.session_state:
        st.session_state["baseline"] = dict(weather)

    st.divider()
    target_class = st.selectbox(
        "Threshold to analyse",
        options=[1, 2],
        index=1,
        format_func=lambda c: f"Transition into {config.RISK_CLASSES[c]}",
    )
    st.caption(f"Factor analysis via {explainer_kind()}.")


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------
profile = get_area_profile(city, area)
record = build_record(city, area, weather, profile)
rkey = _key(record)

bundle = load_model()
prediction = cached_prediction(rkey)
plaus = plausibility_check(city, weather["rainfall_mm"], weather["duration_hr"])
factors = cached_factors(rkey)
band = cached_band(rkey, target_class, "rainfall_mm")

risk_class = prediction["risk_class"]
color = config.RISK_COLORS[risk_class]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(f"### {area}, {city} — City Impact Analysis")
st.caption(str(profile["notes"]))

left, right = st.columns([3, 2])

with left:
    # NOTE: this is raw tree-vote agreement, not a calibrated probability
    st.markdown(
        f"""
        <div class="risk-banner" style="background:{color}">
          <h2>PREDICTED IMPACT: {prediction['risk_label']} RISK</h2>
          <p>Confidence {prediction['confidence']:.0%} (tree-vote agreement) &nbsp;·&nbsp;
             {plaus.badge}</p>
          <p>{prediction['recommended_posture']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("P(LOW)", f"{prediction.get('p_low', 0):.0%}")
    c2.metric("P(MODERATE)", f"{prediction.get('p_moderate', 0):.0%}")
    c3.metric("P(HIGH)", f"{prediction.get('p_high', 0):.0%}")
    st.caption(prediction["confidence_note"])

with right:
    st.markdown("**Spatial vulnerability** &nbsp;<span class='chip chip-locked'>🔒 immutable"
                "</span>", unsafe_allow_html=True)
    spatial_rows = [
        {"Factor": config.FEATURE_LABELS[f], "Value": float(profile[f])}
        for f in config.IMMUTABLE_FEATURES
    ]
    st.dataframe(pd.DataFrame(spatial_rows), hide_index=True,
                 use_container_width=True, height=220)
    st.markdown(
        "<span class='small-note'>Frozen throughout. These describe the ground, "
        "not a lever available tonight.</span>", unsafe_allow_html=True,
    )

st.divider()

tabs = st.tabs([
    "🔍 Factors",
    "🎯 Threshold & recourse",
    "🗺️ Risk surface",
    "🧪 What-if",
    "📝 Decision brief",
    "✅ Feedback",
    "📦 Model card",
])

# --------------------------------------------------------------------------
# Tab 1 — Factors
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Why this risk class?")
    st.write(narrate(factors))
    st.plotly_chart(factor_figure(factors), use_container_width=True)

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("**Grouped attribution**")
        groups = group_contributions(factors)
        groups["share"] = (groups["share"] * 100).round(1)
        st.dataframe(
            groups.rename(columns={"group": "Group", "contribution": "Net effect",
                                   "share": "Share of |effect| (%)"}),
            hide_index=True, use_container_width=True,
        )
    with col_b:
        st.markdown("**Per-feature detail**")
        table = factors[["label", "value", "contribution", "mutability"]].head(10)
        st.dataframe(
            table.rename(columns={"label": "Factor", "value": "Value",
                                  "contribution": "Contribution",
                                  "mutability": "Mutability"}),
            hide_index=True, use_container_width=True,
        )
    st.caption(
        f"Method: {explainer_kind()}. Blue factors are forecastable; purple factors are "
        "terrain and are excluded from the recourse search by design."
    )

# --------------------------------------------------------------------------
# Tab 2 — Threshold & recourse
# --------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"#### Counterfactual threshold range → "
                f"{config.RISK_CLASSES[target_class]}")
    st.info(band.describe())
    st.plotly_chart(threshold_curve_figure(band, record), use_container_width=True)
    st.caption(
        "The band is where the forest shifts from mostly-below to mostly-above the "
        "threshold (35% → 65% of trees). Reported as a range because a single "
        "cut-off would imply precision the model does not have."
    )

    st.markdown("#### Diverse actionable recourse")
    st.caption(
        "Minimal combinations of forecastable conditions that would produce "
        f"{config.RISK_CLASSES[target_class]}. Search space: "
        + ", ".join(config.FEATURE_LABELS[f] for f in SEARCH_FEATURES)
        + ". Spatial features are frozen."
    )

    with st.spinner("Searching actionable space…"):
        cfs = cached_counterfactuals(rkey, city, target_class)

    if cfs["accepted"]:
        rows = [
            {
                "Scenario": describe_counterfactual(c),
                "Plausibility": c["plausibility"].badge,
                "Model confidence": f"{c['confidence']:.0%}",
                "Change cost": round(c["cost"], 3),
            }
            for c in cfs["accepted"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.warning(
            f"No plausible combination of forecastable variables reaches "
            f"{config.RISK_CLASSES[target_class]} for this area. "
            "That is itself a finding — under realistic weather this area does not "
            "cross that threshold."
        )

    st.markdown("**Filtered out by the plausibility check**")
    if cfs["rejected"]:
        for r in cfs["rejected"]:
            st.error(
                f"{describe_counterfactual(r)} — {r['plausibility'].badge}\n\n"
                + " ".join(r["plausibility"].reasons)
            )
    else:
        ex = rejected_example(city)
        st.error(f"Example: {ex['scenario']} — {ex['verdict']}\n\n" + " ".join(ex["reasons"]))
    st.caption(
        f"Searched {cfs['n_candidates']:,} candidate scenarios; "
        f"{cfs['n_reaching_target']:,} reached the target class before plausibility "
        "filtering."
    )

# --------------------------------------------------------------------------
# Tab 3 — Risk surface
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Risk response surface")
    c1, c2 = st.columns(2)
    x_feature = c1.selectbox(
        "Horizontal axis", config.ACTIONABLE_FEATURES, index=0,
        format_func=lambda f: config.FEATURE_LABELS[f],
    )
    y_options = [f for f in config.ACTIONABLE_FEATURES if f != x_feature]
    y_feature = c2.selectbox(
        "Vertical axis", y_options,
        index=y_options.index("duration_hr") if "duration_hr" in y_options else 0,
        format_func=lambda f: config.FEATURE_LABELS[f],
    )

    with st.spinner("Computing response surface…"):
        surface = cached_surface(rkey, x_feature, y_feature, target_class)

    st.plotly_chart(surface_figure(surface, record, city), use_container_width=True)
    st.info(surface_summary(surface, record))
    st.caption(
        "Dashed black line: the 50% decision boundary. Dotted line: the edge of what "
        "this city has ever recorded — everything past it is model extrapolation, "
        "not forecastable reality."
    )

# --------------------------------------------------------------------------
# Tab 4 — What-if
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown("#### What-if simulator")
    st.caption(
        "Adjust the sliders in the sidebar and compare against the baseline forecast "
        "captured when you opened this area."
    )

    baseline = st.session_state["baseline"]
    base_record = build_record(city, area, baseline, profile)
    base_pred = cached_prediction(_key(base_record))

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.markdown("**Baseline forecast**")
        st.metric("Risk", base_pred["risk_label"], f"{base_pred['confidence']:.0%} agreement")
    with c2:
        st.markdown("**Current scenario**")
        delta = prediction["risk_class"] - base_pred["risk_class"]
        st.metric(
            "Risk", prediction["risk_label"],
            f"{prediction['confidence']:.0%} agreement",
            delta_color="inverse" if delta else "off",
        )
    with c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        def _set_preset_values(overrides: dict) -> None:
            for feat, value in overrides.items():
                st.session_state[f"w_{feat}"] = float(value)

        st.button(
            "Reset to baseline",
            use_container_width=True,
            on_click=_set_preset_values,
            args=(baseline,),
        )

    comparison = pd.DataFrame(
        {
            "Variable": [config.FEATURE_LABELS[f] for f in config.ACTIONABLE_FEATURES],
            "Baseline": [baseline[f] for f in config.ACTIONABLE_FEATURES],
            "Current": [weather[f] for f in config.ACTIONABLE_FEATURES],
        }
    )
    comparison["Δ"] = (comparison["Current"] - comparison["Baseline"]).round(1)
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("**Quick scenarios**")
    q1, q2, q3, q4 = st.columns(4)
    presets = {
        "Dry spell": {"rainfall_mm": 5.0, "duration_hr": 6.0},
        "Heavy (IMD)": {"rainfall_mm": 80.0, "duration_hr": 24.0},
        "Very heavy": {"rainfall_mm": 150.0, "duration_hr": 24.0},
        "Cloudburst": {"rainfall_mm": 90.0, "duration_hr": 3.0},
    }
    for col, (name, overrides) in zip((q1, q2, q3, q4), presets.items()):
        col.button(
            name,
            use_container_width=True,
            on_click=_set_preset_values,
            args=(overrides,),
        )

    st.caption(
        f"Plausibility of the current scenario: {plaus.badge} — "
        + " ".join(plaus.reasons)
    )

# --------------------------------------------------------------------------
# Tab 5 — Decision brief
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("#### Historically-grounded decision brief")
    incidents = get_incidents(city, area)

    with st.expander("Grounding records used (real reported events)", expanded=False):
        st.dataframe(
            incidents[["event_date", "rainfall_mm", "duration_hr",
                       "observed_impact", "source_status"]],
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "Records are marked `illustrative-verify` — confirm each against its source "
            "before presenting. The brief is instructed to describe them as reported "
            "events rather than established fact."
        )

    if st.button("Generate decision brief", type="primary"):
        payload = build_payload(
            role=role, city=city, area=area, profile=profile, weather=weather,
            prediction=prediction, factors=factors, band=band,
            counterfactuals=cached_counterfactuals(rkey, city, target_class),
            plausibility=plaus, incidents=incidents,
        )
        shap_summary = narrate(factors)
        threshold_range = band.describe()
        historical_incidents_list = incidents.to_dict(orient="records")

        with st.spinner("Writing brief…"):
            text, used_fallback = generate_decision_brief(
                risk_class=prediction["risk_label"],
                confidence=prediction["confidence"],
                shap_summary=shap_summary,
                threshold_range=threshold_range,
                historical_incidents=historical_incidents_list,
                payload=payload,
            )
        st.session_state["brief"] = {
            "text": text,
            "used_fallback": used_fallback,
            "payload": payload,
        }

    if "brief" in st.session_state:
        brief = st.session_state["brief"]
        if brief.get("used_fallback"):
            st.info("ℹ️ *Note: Fallback template was used because the LLM API was unavailable.*")
        st.markdown(brief["text"])
        with st.expander("Exact payload sent to the model"):
            st.json(brief["payload"])
        st.download_button(
            "Download brief (markdown)", brief["text"],
            file_name=f"weathermind_brief_{city}_{area}.md".replace(" ", "_"),
            mime="text/markdown",
        )
    else:
        st.info(
            "The brief is generated on demand. Without an API key it falls back to a "
            "deterministic template built from the same payload — so the demo never "
            "depends on a network call."
        )

# --------------------------------------------------------------------------
# Tab 6 — Feedback
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("#### Officer feedback → recalibration")
    st.caption(
        "Confirmed outcomes are logged with the full input vector. These become the "
        "locally-observed labels that eventually replace generated ones."
    )

    with st.form("feedback_form"):
        verdict = st.radio(
            f"Was **{prediction['risk_label']}** the right call for {area}?",
            ["Accurate", "Too high", "Too low"], horizontal=True,
        )
        observed = st.text_input("Observed outcome (optional)",
                                 placeholder="e.g. knee-deep water at the junction for 4 hours")
        notes = st.text_area("Notes (optional)", height=80)
        if st.form_submit_button("Log feedback", type="primary"):
            log_feedback(
                role=role, city=city, area=area, weather=weather,
                prediction=prediction, verdict=verdict,
                observed_outcome=observed, notes=notes,
            )
            st.success("Logged to logs/officer_feedback.csv")

    summary = feedback_summary()
    if summary["n"]:
        st.metric("Logged assessments", summary["n"],
                  f"{summary['agreement_rate']:.0%} rated accurate")
    st.caption(
        "Intended loop (not built in the prototype): once an area accumulates enough "
        "confirmed outcomes, those rows are weighted into retraining, and per-area "
        "calibration curves replace the global thresholds."
    )

# --------------------------------------------------------------------------
# Tab 7 — Model card
# --------------------------------------------------------------------------
with tabs[6]:
    metrics = bundle.metrics
    st.markdown("#### Model card")
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Training rows", f"{metrics['n_train']:,}")
    c3.metric("Trees", f"{bundle.model.n_estimators}")

    st.markdown("**Per-class performance (held-out)**")
    report = pd.DataFrame(metrics["report"]).T
    st.dataframe(report.round(3), use_container_width=True)

    st.markdown("**Global feature importance**")
    importance = (
        pd.Series(metrics["feature_importance"]).sort_values(ascending=False).to_frame("importance")
    )
    importance.index = [config.FEATURE_LABELS.get(i, i) for i in importance.index]
    st.bar_chart(importance)

    base = get_baseline(city)
    st.markdown(f"**{city} rainfall baseline used by the plausibility filter**")
    st.dataframe(base.to_frame().T, hide_index=True, use_container_width=True)

    st.warning(
        "Known limitations, stated plainly: training labels come from an explicit "
        "hydrological score rather than observed flood outcomes; the spatial table is "
        "hand-curated from published reports; historical incidents are marked "
        "unverified; and river-stage and reservoir-release effects are not modelled, "
        "which matters most for the riverside areas in Erode and Tirupur."
    )
    st.caption(f"Model trained at {bundle.trained_at} · {explainer_kind()}")
