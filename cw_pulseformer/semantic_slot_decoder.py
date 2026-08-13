"""Deterministic value constraints layered after neural semantic recognition."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .semantic_data import (
    ANTENNA_VALUES,
    FICTIONAL_LOCATIONS,
    FICTIONAL_NAMES,
    RIG_VALUES,
    WEATHER_VALUES,
)


def _catalog_pattern(values: Sequence[str]) -> re.Pattern[str]:
    choices = "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))
    return re.compile(rf"(?<![A-Z0-9])(?:{choices})(?![A-Z0-9])")


DEFAULT_SLOT_CATALOGS = {
    "NAME": FICTIONAL_NAMES,
    "LOCATION": FICTIONAL_LOCATIONS,
    "WEATHER": WEATHER_VALUES,
    "RIG": RIG_VALUES,
    "ANTENNA": ANTENNA_VALUES,
    "REGION": ("JA", "EU", "NA", "SA", "AF", "OC"),
    "CQ_SCOPE": ("DX", "TEST", "QRP"),
}

AGE_VALUE = r"(?:[89]|[1-7][0-9]|8[0-5])"
AGE_PATTERNS = (
    re.compile(rf"(?<![A-Z0-9])AGE(?: IS)? (?P<value>{AGE_VALUE})(?![0-9])"),
    re.compile(rf"(?<![A-Z0-9])I AM (?P<value>{AGE_VALUE}) YEARS OLD(?![A-Z0-9])"),
    re.compile(rf"^(?P<value>{AGE_VALUE})(?: K|$)"),
)

SLOT_PATTERNS = {
    "CALLSIGN": re.compile(
        r"(?<![A-Z0-9])(?=[A-Z0-9]{3,7}(?![A-Z0-9]))(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]+",
    ),
    "NAME": _catalog_pattern(FICTIONAL_NAMES),
    "AGE": re.compile(rf"(?<![0-9]){AGE_VALUE}(?![0-9])"),
    "LOCATION": _catalog_pattern(FICTIONAL_LOCATIONS),
    "WEATHER": _catalog_pattern(WEATHER_VALUES),
    "POWER": re.compile(r"(?<![A-Z0-9])(?:1|2|5|10|20|50|100)W(?![A-Z0-9])"),
    "RIG": _catalog_pattern(RIG_VALUES),
    "ANTENNA": _catalog_pattern(ANTENNA_VALUES),
    "RST": re.compile(r"(?<![0-9])[1-5][1-9][1-9](?![0-9])"),
    "REGION": _catalog_pattern(DEFAULT_SLOT_CATALOGS["REGION"]),
    "CQ_SCOPE": _catalog_pattern(DEFAULT_SLOT_CATALOGS["CQ_SCOPE"]),
}

CRITICAL_SLOT_PATTERNS = {
    name: SLOT_PATTERNS[name] for name in ("CALLSIGN", "RST", "POWER")
}


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _age_values(text: str) -> tuple[str, ...]:
    return _unique(tuple(
        match.group("value")
        for pattern in AGE_PATTERNS
        for match in pattern.finditer(text)
    ))


def decode_slot_values(
    text: str,
    topic_scores: Mapping[str, float | bool],
    *,
    safe_to_commit: bool,
    threshold: float = 0.5,
    catalogs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return grammar-valid or catalog-valid values after the neural safety gate."""
    if not safe_to_commit:
        return {}
    normalized = " ".join(str(text).upper().split())
    patterns = dict(SLOT_PATTERNS)
    if catalogs:
        for name, values in catalogs.items():
            if values:
                patterns[name] = _catalog_pattern(tuple(str(value).upper() for value in values))
    decoded: dict[str, tuple[str, ...]] = {}
    for name, pattern in patterns.items():
        if float(topic_scores.get(name, 0.0)) < threshold:
            continue
        if name == "AGE":
            values = _age_values(normalized)
        else:
            values = _unique(tuple(match.group(0) for match in pattern.finditer(normalized)))
        if values:
            decoded[name] = values
    return decoded


def decode_critical_slot_values(
    text: str,
    topic_scores: Mapping[str, float | bool],
    *,
    safe_to_commit: bool,
    threshold: float = 0.5,
) -> dict[str, tuple[str, ...]]:
    decoded = decode_slot_values(
        text, topic_scores, safe_to_commit=safe_to_commit, threshold=threshold,
    )
    return {name: decoded[name] for name in CRITICAL_SLOT_PATTERNS if name in decoded}


def labeled_slot_values(
    text: str,
    labels: Sequence[int],
    label_id: int,
) -> tuple[str, ...]:
    """Decode contiguous ground-truth spans; labels include the leading CLS row."""
    values: list[str] = []
    current = ""
    for index, character in enumerate(text):
        if index + 1 < len(labels) and int(labels[index + 1]) == label_id:
            current += character
        elif current:
            values.append(current)
            current = ""
    if current:
        values.append(current)
    return _unique(tuple(values))
