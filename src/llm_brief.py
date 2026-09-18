"""
WeatherMind — grounded decision brief.

The LLM is deliberately the LAST component and the least trusted one. It never
decides anything. It receives a structured payload — model output, confidence,
factor analysis, counterfactual thresholds, plausibility verdict and real
historical incidents for that exact area — and is asked to turn it into a brief
an officer can act on, using nothing beyond what it was given.

If no API key is configured, `generate_brief` falls back to a deterministic
template built from the same payload. The demo therefore never depends on a
network call, and you can show the judges both versions side by side.

Configure a key either in `.streamlit/secrets.toml`:

    ANTHROPIC_API_KEY = "sk-ant-..."

or as an environment variable of the same name.
"""

from __future__ import annotations

import json
import os
import textwrap

from . import config
from .counterfactual import describe_counterfactual

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are WeatherMind's decision-brief writer, supporting Indian civic and
    emergency officials during rainfall events.

    Hard rules:
    - Use ONLY the facts in the payload. Never invent rainfall figures, dates,
      incidents, casualty numbers or place details.
    - The risk class comes from the model. Do not overrule it, soften it, or
      escalate it on your own.
    - Report confidence honestly. If tree agreement is low, say the call is
      borderline and lean on the threshold range instead of the label.
    - Spatial vulnerability is fixed context, not a recommendation target. Never
      suggest that drainage or elevation be changed to manage tonight's event.
    - If a historical incident is marked unverified, refer to it as a reported
      event, not established fact.

    Format (markdown, no preamble):
    **Situation** - two sentences.
    **Assessment** - risk class, confidence, and the dominant drivers.
    **Threshold to watch** - the counterfactual range, framed as a trigger.
    **Historical precedent** - what comparable past events did in this area.
    **Recommended actions** - 3 to 5 bullets, specific to the stated role.
    **Confidence caveat** - one line on what would change this assessment.

    Keep the whole brief under 320 words. Plain operational English.
    """
).strip()


def build_payload(
    *,
    role: str,
    city: str,
    area: str,
    profile,
    weather: dict,
    prediction: dict,
    factors,
    band,
    counterfactuals: dict,
    plausibility,
    incidents,
) -> dict:
    """Assemble everything the brief is allowed to know."""
    top_factors = [
        {
            "factor": r.label,
            "value": round(float(r.value), 2),
            "effect": r.direction,
            "mutability": r.mutability,
        }
        for r in factors.head(5).itertuples()
    ]

    cf_list = [
        {
            "change": describe_counterfactual(c),
            "resulting_class": counterfactuals["target_label"],
            "plausibility": c["plausibility"].level,
        }
        for c in counterfactuals.get("accepted", [])[:3]
    ]

    incident_list = [
        {
            "date": r.event_date,
            "rainfall_mm": float(r.rainfall_mm),
            "duration_hr": float(r.duration_hr),
            "impact": r.observed_impact,
            "response": r.response_note,
            "verification": r.source_status,
        }
        for r in incidents.itertuples()
    ]

    return {
        "role": role,
        "role_focus": config.ROLES.get(role, ""),
        "location": {"city": city, "area": area, "context_note": str(profile["notes"])},
        "spatial_context_immutable": {
            config.FEATURE_LABELS[f]: float(profile[f]) for f in config.IMMUTABLE_FEATURES
        },
        "forecast_inputs": {
            config.FEATURE_LABELS[f]: round(float(v), 1) for f, v in weather.items()
        },
        "model_output": {
            "risk_class": prediction["risk_label"],
            "tree_vote_agreement": round(prediction["confidence"], 3),
            "class_probabilities": {
                k.replace("p_", ""): round(float(v), 3)
                for k, v in prediction.items()
                if k.startswith("p_")
            },
            "standard_posture": prediction["recommended_posture"],
        },
        "top_factors": top_factors,
        "threshold_band": {
            "variable": config.FEATURE_LABELS[band.feature],
            "target_class": config.RISK_CLASSES[band.target_class],
            "lower": None if band.lower is None else round(band.lower, 1),
            "midpoint": None if band.central is None else round(band.central, 1),
            "upper": None if band.upper is None else round(band.upper, 1),
            "current_value": round(band.current_value, 1),
            "note": band.describe(),
        },
        "actionable_counterfactuals": cf_list,
        "plausibility_of_current_forecast": {
            "level": plausibility.level,
            "imd_band": plausibility.imd_band,
            "reasons": plausibility.reasons,
        },
        "historical_incidents": incident_list,
    }


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
def _get_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:  # pragma: no cover
        import streamlit as st

        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None


def _call_anthropic(payload: dict, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=_get_secret("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": "Write the decision brief from this payload:\n\n"
                + json.dumps(payload, indent=2, default=str),
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


# --------------------------------------------------------------------------
# Deterministic fallback & Decision Brief generator
# --------------------------------------------------------------------------
def generate_fallback_brief(
    risk_class,
    confidence,
    threshold_range,
    historical_incidents,
    payload: dict | None = None,
) -> str:
    """Simple templated brief used if the LLM API fails or is unconfigured."""
    if payload:
        return template_brief(payload)

    incident_note = ""
    if historical_incidents:
        first_inc = historical_incidents[0]
        if isinstance(first_inc, dict):
            date = first_inc.get("date") or first_inc.get("event_date") or "past date"
            impact = first_inc.get("impact") or first_inc.get("observed_impact") or "incident reported"
            incident_note = f" Similar conditions caused incidents here before ({date}: {impact})."
        else:
            incident_note = f" Similar conditions caused incidents here before: {first_inc}."

    conf_pct = f"{confidence:.0%}" if isinstance(confidence, (float, int)) else str(confidence)
    return (
        f"Risk Level: {risk_class} ({conf_pct} confidence). "
        f"Risk class would shift at a rainfall threshold of {threshold_range}."
        f"{incident_note} Recommend monitoring conditions closely."
    )


def generate_decision_brief(
    risk_class,
    confidence,
    shap_summary,
    threshold_range,
    historical_incidents,
    payload: dict | None = None,
    model: str = "claude-3-5-sonnet-20241022",
) -> tuple[str, bool]:
    """
    Generate a decision brief using Claude (Anthropic) as the primary provider.
    Wrapped in try/except so if it fails or times out, the app falls back to a
    templated brief instead of crashing.
    Returns (brief_text, used_fallback: bool).
    """
    conf_pct = f"{confidence:.0%}" if isinstance(confidence, (float, int)) else str(confidence)
    prompt = f"""Generate a concise, actionable decision brief for an emergency responder.

Risk Level: {risk_class}
Confidence: {conf_pct}
Key contributing factors: {shap_summary}
Rainfall threshold for risk transition: {threshold_range}
Historical incidents at this location: {historical_incidents}

Write 3-4 sentences: what the risk means, why (factors), and one recommended action."""

    api_key = _get_secret("ANTHROPIC_API_KEY")

    if api_key:
        try:
            from anthropic import Anthropic  # Primary provider: Claude
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text, False
        except Exception as e:
            print(f"LLM call failed: {e}. Falling back to template.")

    fallback_text = generate_fallback_brief(
        risk_class, confidence, threshold_range, historical_incidents, payload
    )
    return fallback_text, True


def template_brief(payload: dict) -> str:
    m = payload["model_output"]
    band = payload["threshold_band"]
    loc = payload["location"]
    drivers = ", ".join(
        f"{f['factor']} ({f['value']}, {f['effect']})" for f in payload["top_factors"][:3]
    )
    incidents = payload["historical_incidents"]

    if band["midpoint"] is not None and band["current_value"] < band["lower"]:
        trigger = (
            f"{band['variable']} is currently {band['current_value']}. "
            f"The shift into {band['target_class']} occurs between {band['lower']} "
            f"and {band['upper']} (midpoint {band['midpoint']}). Treat "
            f"{band['lower']} as the escalation trigger — roughly "
            f"{band['lower'] - band['current_value']:.0f} of headroom."
        )
    elif band["midpoint"] is not None:
        trigger = (
            f"{band['variable']} is already at {band['current_value']}, past the "
            f"{band['target_class']} boundary of {band['lower']}–{band['upper']}. "
            f"De-escalation would require the total to come in below about "
            f"{band['lower']}, so plan for {band['target_class']} unless the forecast "
            "is revised down sharply."
        )
    else:
        trigger = (
            f"No value of {band['variable'].lower()} inside the searched range moves "
            f"this area into {band['target_class']} on its own."
        )

    hist = "\n".join(
        f"- {i['date']}: {i['rainfall_mm']:.0f} mm over {i['duration_hr']:.0f} h — "
        f"{i['impact']} (reported; {i['verification']})"
        for i in incidents[:3]
    ) or "- No recorded incidents on file for this area."

    actions = "\n".join(f"- {a}" for a in _role_actions(payload))

    plaus = payload["plausibility_of_current_forecast"]
    caveat = (
        "Agreement is strong; the class is stable against small forecast errors."
        if m["tree_vote_agreement"] >= 0.75
        else "Agreement is weak — this scenario sits near a decision boundary. "
             "Act on the threshold range rather than the label."
    )

    lines = [
        "**Situation**",
        f"{loc['area']}, {loc['city']} is forecast to receive "
        f"{payload['forecast_inputs'].get('Rainfall (mm)')} mm over "
        f"{payload['forecast_inputs'].get('Event duration (hr)')} hours "
        f"({plaus['imd_band'].lower()}; {plaus['level']} for this city). "
        f"{loc['context_note']}",
        "",
        "**Assessment**",
        f"Predicted risk: **{m['risk_class']}** with "
        f"{m['tree_vote_agreement']:.0%} tree agreement. "
        f"Dominant drivers: {drivers}.",
        "",
        "**Threshold to watch**",
        trigger,
        "",
        "**Historical precedent**",
        hist,
        "",
        "**Recommended actions**",
        actions,
        "",
        "**Confidence caveat**",
        caveat,
        "",
        "_Generated without an LLM (no API key configured). Every figure above comes "
        "directly from the model pipeline._",
    ]
    return "\n".join(lines)


def _role_actions(payload: dict) -> list:
    cls = payload["model_output"]["risk_class"]
    role = payload["role"]
    area = payload["location"]["area"]

    base = {
        "LOW": [
            "Maintain routine monitoring; no pre-positioning required.",
            "Re-run the assessment if the forecast total revises upward.",
        ],
        "MODERATE": [
            f"Place local teams on standby for {area}.",
            "Verify pump readiness and clear known drain choke points.",
            "Brief field units on the escalation trigger stated above.",
        ],
        "HIGH": [
            f"Pre-position rescue assets near {area} before rainfall peaks.",
            "Issue an area-level public advisory.",
            "Prepare diversion for low-lying road stretches and underpasses.",
            "Open communication with the district control room.",
        ],
    }[cls]

    if role.startswith("Police"):
        base.append("Assign traffic personnel to junctions that historically flood first.")
    elif role.startswith("Emergency"):
        base.append("Confirm boat, pump and lighting availability at the nearest depot.")
    elif role.startswith("Municipal"):
        base.append("Deploy desilting crews to the area's primary stormwater outfall.")
    else:
        base.append("Confirm relief-centre readiness and inter-agency contact list.")
    return base


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def generate_brief(payload: dict, provider: str = "auto",
                   model: str | None = None) -> tuple[str, str]:
    """Return (brief_markdown, source_label)."""
    model = model or config.DEFAULT_LLM_MODEL
    m = payload["model_output"]
    band = payload["threshold_band"]

    text, used_fallback = generate_decision_brief(
        risk_class=m["risk_class"],
        confidence=m["tree_vote_agreement"],
        shap_summary=", ".join(f"{f['factor']} ({f['effect']})" for f in payload.get("top_factors", [])[:3]),
        threshold_range=band.get("note", ""),
        historical_incidents=payload.get("historical_incidents", []),
        payload=payload,
        model=model,
    )
    if not used_fallback:
        return text, f"Claude ({model})"
    return text, "Deterministic template (no API key)"
