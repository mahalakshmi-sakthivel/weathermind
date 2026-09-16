"""
WeatherMind — plausibility filter.

A counterfactual is useless if it is meteorologically absurd. "This area would
be safe if only 900 mm fell in one hour instead of 60" is technically a valid
model inversion and operationally worthless.

Every candidate scenario is therefore checked against the city's observed
rainfall distribution (`data/rainfall_baselines.csv`) on two axes — total depth
and intensity — before it is ever shown to an officer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .config import imd_band
from .data_loader import get_baseline


@dataclass
class Plausibility:
    plausible: bool
    level: str            # "typical" | "uncommon" | "rare" | "unprecedented"
    daily_equivalent_mm: float
    intensity_mm_per_hr: float
    imd_band: str
    reasons: list

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def badge(self) -> str:
        return {
            "typical": "🟢 Typical for this city",
            "uncommon": "🟡 Uncommon but observed",
            "rare": "🟠 Rare — near the historical extreme",
            "unprecedented": "🔴 Beyond anything on record — rejected",
        }[self.level]


def check(city: str, rainfall_mm: float, duration_hr: float) -> Plausibility:
    """Classify how realistic a rainfall scenario is for a given city."""
    base = get_baseline(city)
    duration_hr = max(float(duration_hr), 0.5)

    # Compare like with like: a 48 h total is spread over two days.
    days = max(duration_hr / 24.0, 1.0)
    daily_equiv = float(rainfall_mm) / days
    intensity = float(rainfall_mm) / duration_hr

    p90 = float(base["daily_p90_mm"])
    p99 = float(base["daily_p99_mm"])
    record = float(base["daily_record_mm"])
    max_intensity = float(base["max_intensity_mm_per_hr"])

    reasons = []
    level = "typical"

    if daily_equiv > record * 1.1:
        level = "unprecedented"
        reasons.append(
            f"Daily-equivalent depth {daily_equiv:.0f} mm exceeds the {city} "
            f"record of about {record:.0f} mm."
        )
    elif daily_equiv > p99:
        level = "rare"
        reasons.append(
            f"{daily_equiv:.0f} mm/day sits above the 99th percentile "
            f"(~{p99:.0f} mm) for {city}."
        )
    elif daily_equiv > p90:
        level = "uncommon"
        reasons.append(
            f"{daily_equiv:.0f} mm/day is above the 90th percentile "
            f"(~{p90:.0f} mm) — a heavy but regularly observed spell."
        )
    else:
        reasons.append(f"{daily_equiv:.0f} mm/day is within the normal wet-season range.")

    if intensity > max_intensity * 1.25:
        level = "unprecedented"
        reasons.append(
            f"Intensity of {intensity:.1f} mm/hr exceeds the highest short-duration "
            f"rate recorded locally (~{max_intensity:.0f} mm/hr)."
        )
    elif intensity > max_intensity * 0.85 and level in ("typical", "uncommon"):
        level = "rare"
        reasons.append(f"Intensity of {intensity:.1f} mm/hr approaches the local maximum.")

    return Plausibility(
        plausible=(level != "unprecedented"),
        level=level,
        daily_equivalent_mm=round(daily_equiv, 1),
        intensity_mm_per_hr=round(intensity, 2),
        imd_band=imd_band(daily_equiv),
        reasons=reasons,
    )


def rejected_example(city: str) -> dict:
    """A deliberately implausible scenario, for the 'what we filter out' panel."""
    base = get_baseline(city)
    absurd_rain = float(base["daily_record_mm"]) * 2.0
    result = check(city, absurd_rain, 6.0)
    return {
        "scenario": f"{absurd_rain:.0f} mm in 6 hours",
        "verdict": result.badge,
        "reasons": result.reasons,
    }
