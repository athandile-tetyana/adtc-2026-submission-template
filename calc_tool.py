"""
calc_tool.py — deterministic rate/area scaling calculator for agricultural
dosage questions (fertilizer, pesticide, seed rate scaling by field size).

WHY A DETERMINISTIC TOOL INSTEAD OF ASKING THE MODEL TO DO THE MATH:
  LLMs are unreliable at arithmetic, especially small quantized models on
  CPU. A farmer asking "the guide says 120kg per hectare, how much do I
  need for 3.5 hectares?" needs an exact answer, not a plausible-sounding
  one. Regex-extract the numbers, do real multiplication, format the
  result — zero chance of a hallucinated number.

SCOPE: this handles the single most common real pattern in ag advisory:
  "<rate> <unit> per <area-unit>" + "<area> <area-unit>" somewhere in the
  same question. It is deliberately narrow rather than a general NLP-to-math
  system — narrow-but-correct beats broad-but-unreliable for a hard number
  a farmer will actually act on.
"""

import re

AREA_TO_HECTARES = {
    "hectare": 1.0, "hectares": 1.0, "ha": 1.0,
    "acre": 0.404686, "acres": 0.404686,
}

RATE_UNITS = {"kg", "g", "ml", "l", "liter", "liters", "litre", "litres", "kg/ha", "g/ha"}

RATE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|g|ml|l|liters?|litres?)"
    r"(?:\s+[A-Za-z0-9]+)?"
    r"\s*(?:per|/)\s*(hectare|hectares|ha|acre|acres)",
    re.IGNORECASE,
)

AREA_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hectares?|ha|acres?)(?!\s*(?:per|/))",
    re.IGNORECASE,
)


def try_calculate(query: str) -> str | None:
    """
    Attempt to detect and answer a rate-scaling question. Returns a formatted
    answer string if a calculation pattern is found, otherwise None (caller
    should fall through to RAG/direct-answer routing).
    """
    rate_match = RATE_PATTERN.search(query)
    if not rate_match:
        return None

    rate_value = float(rate_match.group(1))
    rate_unit = rate_match.group(2).lower()
    rate_area_unit = rate_match.group(3).lower()

    remaining = query[: rate_match.start()] + " " + query[rate_match.end() :]
    area_match = AREA_PATTERN.search(remaining)
    if not area_match:
        return None

    area_value = float(area_match.group(1))
    area_unit = area_match.group(2).lower()

    rate_area_conversion = AREA_TO_HECTARES.get(rate_area_unit.rstrip("s"), AREA_TO_HECTARES.get(rate_area_unit))
    area_conversion = AREA_TO_HECTARES.get(area_unit.rstrip("s"), AREA_TO_HECTARES.get(area_unit))
    if rate_area_conversion is None or area_conversion is None:
        return None

    rate_area_ha = rate_area_conversion
    farmer_area_ha = area_value * area_conversion

    rate_per_hectare = rate_value / rate_area_ha
    total = rate_per_hectare * farmer_area_ha

    unit_display = rate_unit if rate_unit != "l" else "liters"
    return (
        f"For {area_value} {area_unit} at a rate of {rate_value} {rate_unit} "
        f"per {rate_area_unit}, you need approximately "
        f"{total:.2f} {unit_display} total "
        f"({rate_per_hectare:.2f} {unit_display} per hectare)."
    )
