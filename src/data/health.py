"""EPA AQI health-category mapping. Used in the dashboard and alerts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AqiCategory:
    label: str
    lo: int
    hi: int
    color: str
    advice: str


AQI_CATEGORIES: tuple[AqiCategory, ...] = (
    AqiCategory("Good",                            0,   50,  "#009966",
                "Air quality is satisfactory."),
    AqiCategory("Moderate",                       51,  100,  "#ffde33",
                "Air quality is acceptable. Unusually sensitive people should limit prolonged outdoor exertion."),
    AqiCategory("Unhealthy for Sensitive Groups",101,  150,  "#ff9933",
                "Sensitive groups should reduce prolonged outdoor exertion."),
    AqiCategory("Unhealthy",                     151,  200,  "#cc0033",
                "Everyone should limit outdoor exertion. Sensitive groups should avoid it."),
    AqiCategory("Very Unhealthy",                201,  300,  "#660099",
                "Health alert: everyone may experience serious effects. Avoid outdoor exertion."),
    AqiCategory("Hazardous",                     301, 10000, "#7e0023",
                "Health warning of emergency conditions. Everyone should stay indoors."),
)

HAZARDOUS_THRESHOLD = 151  # "Unhealthy" and above triggers dashboard alerts


def category_for(aqi: float) -> AqiCategory:
    for cat in AQI_CATEGORIES:
        if cat.lo <= aqi <= cat.hi:
            return cat
    return AQI_CATEGORIES[-1]


def is_hazardous(aqi: float) -> bool:
    return aqi >= HAZARDOUS_THRESHOLD
