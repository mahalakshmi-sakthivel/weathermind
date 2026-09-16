"""
WeatherMind — central configuration.

Everything that the rest of the codebase needs to agree on lives here:
paths, the feature taxonomy (actionable vs immutable), risk classes,
and the city/area registry.

The actionable/immutable split is the conceptual heart of the project:
the counterfactual engine is only ever allowed to move ACTIONABLE_FEATURES.
IMMUTABLE_FEATURES describe the ground and are frozen context.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

SPATIAL_CSV = DATA_DIR / "spatial_vulnerability.csv"
INCIDENTS_CSV = DATA_DIR / "historical_incidents.csv"
BASELINES_CSV = DATA_DIR / "rainfall_baselines.csv"
TRAINING_CSV = DATA_DIR / "training_data.csv"

# Drop a real Kaggle export here and the loader will prefer it over the
# bundled generated training set. See README section "Swapping in real data".
KAGGLE_CSV = DATA_DIR / "kaggle_rainfall.csv"

MODEL_PATH = MODEL_DIR / "risk_model.joblib"
FEEDBACK_LOG = LOG_DIR / "officer_feedback.csv"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Feature taxonomy
# --------------------------------------------------------------------------
# Forecastable / controllable in a what-if sense. The counterfactual search
# is restricted to these.
ACTIONABLE_FEATURES = [
    "rainfall_mm",            # total rainfall over the event window
    "duration_hr",            # hours over which that rainfall falls
    "wind_speed_kmph",
    "humidity_pct",
    "temperature_c",
    "pressure_hpa",
    "antecedent_rain_3d_mm",  # rain in the preceding 72 h (soil already wet?)
]

# Properties of the place. Never varied by the counterfactual engine.
IMMUTABLE_FEATURES = [
    "elevation_m",
    "slope_pct",
    "drainage_vulnerability",
    "imperviousness",
    "soil_infiltration",
]

# Derived at feature-engineering time from the above.
DERIVED_FEATURES = [
    "intensity_mm_per_hr",
    "effective_runoff_mm",
    "drainage_deficit",
]

FEATURE_ORDER = ACTIONABLE_FEATURES + IMMUTABLE_FEATURES + DERIVED_FEATURES

# Which actionable features the What-If sliders and the counterfactual
# search are allowed to touch, with UI bounds and a step.
ACTIONABLE_BOUNDS = {
    "rainfall_mm": (0.0, 400.0, 1.0),
    "duration_hr": (1.0, 48.0, 1.0),
    "wind_speed_kmph": (0.0, 120.0, 1.0),
    "humidity_pct": (20.0, 100.0, 1.0),
    "temperature_c": (15.0, 45.0, 0.5),
    "pressure_hpa": (985.0, 1020.0, 0.5),
    "antecedent_rain_3d_mm": (0.0, 300.0, 1.0),
}

# Human-readable labels used across the UI and the LLM prompt.
FEATURE_LABELS = {
    "rainfall_mm": "Rainfall (mm)",
    "duration_hr": "Event duration (hr)",
    "wind_speed_kmph": "Wind speed (km/h)",
    "humidity_pct": "Humidity (%)",
    "temperature_c": "Temperature (°C)",
    "pressure_hpa": "Pressure (hPa)",
    "antecedent_rain_3d_mm": "Antecedent rain, 72 h (mm)",
    "elevation_m": "Elevation (m)",
    "slope_pct": "Slope (%)",
    "drainage_vulnerability": "Drainage vulnerability (0-1)",
    "imperviousness": "Imperviousness (0-1)",
    "soil_infiltration": "Soil infiltration (0-1)",
    "intensity_mm_per_hr": "Rain intensity (mm/hr)",
    "effective_runoff_mm": "Effective runoff (mm)",
    "drainage_deficit": "Drainage deficit",
}

# --------------------------------------------------------------------------
# Risk classes
# --------------------------------------------------------------------------
RISK_CLASSES = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
RISK_COLORS = {0: "#1a9850", 1: "#fdae61", 2: "#d73027"}

RISK_ACTIONS = {
    0: "Routine monitoring. No pre-positioning required.",
    1: "Place local response teams on standby; check pump and drain readiness "
       "at known choke points.",
    2: "Pre-position rescue assets, issue area-level advisory, and prepare "
       "traffic diversion for low-lying stretches.",
}

# --------------------------------------------------------------------------
# Roles (change the framing of the LLM decision brief)
# --------------------------------------------------------------------------
ROLES = {
    "Police / Traffic Control": (
        "Prioritise road closures, traffic diversion, underpass safety and "
        "crowd movement."
    ),
    "Emergency Response (Fire & Rescue / SDRF)": (
        "Prioritise rescue-asset pre-positioning, boat and pump deployment, "
        "and evacuation staging points."
    ),
    "Municipal Corporation": (
        "Prioritise stormwater drain clearance, pumping stations, sanitation "
        "risk and public advisories."
    ),
    "District Administration": (
        "Prioritise inter-agency coordination, relief centre activation and "
        "public communication."
    ),
}

# --------------------------------------------------------------------------
# City / area registry (kept in sync with data/spatial_vulnerability.csv)
# --------------------------------------------------------------------------
CITY_AREAS = {
    "Chennai": ["Velachery", "Mudichur", "Pallikaranai"],
    "Coimbatore": ["Peelamedu", "Ukkadam", "Singanallur"],
    "Erode": ["ABD Road", "Bhavani Riverside", "Kasipalayam"],
    "Tirupur": ["Avinashi Road", "Noyyal Riverside", "Veerapandi"],
}

# IMD daily-rainfall vocabulary — used for plausibility narration.
IMD_RAIN_BANDS = [
    (0.0, 2.4, "No rain / trace"),
    (2.5, 15.5, "Light rain"),
    (15.6, 64.4, "Moderate rain"),
    (64.5, 115.5, "Heavy rain"),
    (115.6, 204.4, "Very heavy rain"),
    (204.5, 10_000.0, "Extremely heavy rain"),
]


def imd_band(rainfall_mm: float) -> str:
    """Return the IMD descriptive band for a 24 h rainfall total."""
    for lo, hi, label in IMD_RAIN_BANDS:
        if lo <= rainfall_mm <= hi:
            return label
    return "Unclassified"


DEFAULT_LLM_MODEL = "claude-sonnet-5"
