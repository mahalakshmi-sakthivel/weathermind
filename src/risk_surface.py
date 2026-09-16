"""
WeatherMind — risk response surface.

A single risk label answers "what happens at tonight's forecast?". The response
surface answers the more useful question: "how does risk behave across the
whole neighbourhood of tonight's forecast?" — which is what tells an officer
whether they are standing on a plateau or on a cliff edge.

Two actionable variables are swept (rainfall × duration by default); everything
else, including the entire spatial layer, is held fixed. The plausibility
boundary is drawn on top so the part of the surface that cannot physically
happen in this city is visibly marked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_loader import get_baseline
from .model import p_at_least


def compute_surface(
    bundle,
    record: dict,
    x_feature: str = "rainfall_mm",
    y_feature: str = "duration_hr",
    target_class: int = 2,
    nx: int = 44,
    ny: int = 32,
) -> dict:
    """Grid P(risk >= target_class) over two actionable features."""
    for f in (x_feature, y_feature):
        if f not in config.ACTIONABLE_FEATURES:
            raise ValueError(f"{f} is immutable and cannot be swept.")

    x_lo, x_hi, _ = config.ACTIONABLE_BOUNDS[x_feature]
    y_lo, y_hi, _ = config.ACTIONABLE_BOUNDS[y_feature]
    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)

    records = [
        dict(record, **{x_feature: float(x), y_feature: float(y)})
        for y in ys
        for x in xs
    ]
    z = p_at_least(bundle, records, target_class).reshape(ny, nx)

    return {
        "x": xs,
        "y": ys,
        "z": z,
        "x_feature": x_feature,
        "y_feature": y_feature,
        "target_class": target_class,
    }


def plausibility_mask(surface: dict, city: str) -> np.ndarray | None:
    """Boolean grid marking scenarios beyond the city's historical record.

    Only meaningful when the swept axes are rainfall and duration.
    """
    if {surface["x_feature"], surface["y_feature"]} != {"rainfall_mm", "duration_hr"}:
        return None

    base = get_baseline(city)
    record_mm = float(base["daily_record_mm"])
    max_int = float(base["max_intensity_mm_per_hr"])

    xs, ys = surface["x"], surface["y"]
    rain = xs if surface["x_feature"] == "rainfall_mm" else ys
    dur = ys if surface["y_feature"] == "duration_hr" else xs

    R, D = np.meshgrid(rain, dur)
    if surface["x_feature"] != "rainfall_mm":
        R, D = D, R

    days = np.maximum(D / 24.0, 1.0)
    return ((R / days) > record_mm * 1.1) | ((R / D) > max_int * 1.25)


def surface_summary(surface: dict, record: dict) -> str:
    """Describe the local gradient — plateau or cliff edge?"""
    xs, z = surface["x"], surface["z"]
    yi = int(np.argmin(np.abs(surface["y"] - record[surface["y_feature"]])))
    row = z[yi]
    xi = int(np.argmin(np.abs(xs - record[surface["x_feature"]])))

    window = slice(max(0, xi - 3), min(len(xs), xi + 4))
    span = row[window].max() - row[window].min()
    x_label = config.FEATURE_LABELS[surface["x_feature"]]
    delta = xs[1] - xs[0]

    if span < 0.10:
        shape = "a flat plateau — small forecast errors barely move the outcome"
    elif span < 0.35:
        shape = "a moderate gradient — the outcome is somewhat sensitive to forecast error"
    else:
        shape = "a steep cliff edge — a small forecast error flips the decision"

    return (
        f"Around the current forecast, risk changes by {span:.0%} across "
        f"±{3 * delta:.0f} of {x_label.lower()}: {shape}."
    )


# --------------------------------------------------------------------------
# Plotly figures (imported lazily so the core stays dependency-light)
# --------------------------------------------------------------------------
def surface_figure(surface: dict, record: dict, city: str | None = None):
    import plotly.graph_objects as go

    x_label = config.FEATURE_LABELS[surface["x_feature"]]
    y_label = config.FEATURE_LABELS[surface["y_feature"]]
    target = config.RISK_CLASSES[surface["target_class"]]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=surface["x"], y=surface["y"], z=surface["z"],
            colorscale="RdYlGn_r", zmin=0, zmax=1,
            colorbar=dict(title=f"P(≥ {target})"),
            hovertemplate=(f"{x_label}: %{{x:.0f}}<br>{y_label}: %{{y:.0f}}"
                           f"<br>P(≥ {target}): %{{z:.0%}}<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Contour(
            x=surface["x"], y=surface["y"], z=surface["z"],
            contours=dict(start=0.5, end=0.5, size=0.1, coloring="none",
                          showlabels=True, labelfont=dict(color="black", size=11)),
            line=dict(color="black", width=2, dash="dash"),
            showscale=False, hoverinfo="skip", name="50% boundary",
        )
    )

    if city:
        mask = plausibility_mask(surface, city)
        if mask is not None and mask.any():
            fig.add_trace(
                go.Contour(
                    x=surface["x"], y=surface["y"], z=mask.astype(float),
                    contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                    line=dict(color="#222", width=1.5, dash="dot"),
                    showscale=False, hoverinfo="skip",
                    name="beyond record",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=[record[surface["x_feature"]]], y=[record[surface["y_feature"]]],
            mode="markers+text", text=["current forecast"], textposition="top center",
            marker=dict(size=14, color="white", line=dict(color="black", width=2),
                        symbol="circle"),
            hovertemplate="Current forecast<extra></extra>", showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Risk response surface — P(risk ≥ {target})",
        xaxis_title=x_label, yaxis_title=y_label,
        height=470, margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def threshold_curve_figure(band, record: dict):
    """The 1-D sweep behind the threshold band, with the uncertainty region."""
    import plotly.graph_objects as go

    curve = band.curve
    feat = band.feature
    label = config.FEATURE_LABELS[feat]
    target = config.RISK_CLASSES[band.target_class]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=curve[feat], y=curve["p_at_least"], mode="lines",
                   line=dict(color="#bbb", width=1), name="raw")
    )
    fig.add_trace(
        go.Scatter(x=curve[feat], y=curve["smoothed"], mode="lines",
                   line=dict(color="#d73027", width=3), name="smoothed")
    )

    if band.found and band.lower is not None and band.upper is not None:
        fig.add_vrect(x0=band.lower, x1=band.upper, fillcolor="#fdae61",
                      opacity=0.35, line_width=0,
                      annotation_text="transition band", annotation_position="top left")
    fig.add_vline(x=record[feat], line=dict(color="black", dash="dot"),
                  annotation_text="current", annotation_position="bottom right")
    fig.add_hline(y=0.5, line=dict(color="#888", dash="dash"))

    fig.update_layout(
        title=f"Probability of reaching {target} as {label.lower()} varies",
        xaxis_title=label, yaxis_title=f"P(risk ≥ {target})",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=380, margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig


def factor_figure(factors: pd.DataFrame, top_n: int = 8):
    """Horizontal bar chart of factor contributions, coloured by mutability."""
    import plotly.graph_objects as go

    top = factors.head(top_n).iloc[::-1]
    colors = ["#2c7fb8" if m == "actionable" else "#756bb1" for m in top["mutability"]]

    fig = go.Figure(
        go.Bar(
            x=top["contribution"], y=top["label"], orientation="h",
            marker_color=colors,
            customdata=np.stack([top["value"], top["mutability"]], axis=-1),
            hovertemplate="%{y}<br>value: %{customdata[0]:.2f}"
                          "<br>contribution: %{x:+.3f}<br>%{customdata[1]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#444", width=1))
    fig.update_layout(
        title="Contribution to the predicted risk class "
              "(blue = forecastable, purple = immutable terrain)",
        xaxis_title="← lowers risk   |   raises risk →",
        height=380, margin=dict(l=10, r=20, t=60, b=50),
    )
    return fig
