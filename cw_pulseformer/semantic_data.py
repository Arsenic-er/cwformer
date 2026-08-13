"""Synthetic, privacy-safe CW QSO semantics with explicit dialogue context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SEMANTIC_SCHEMA_VERSION = "qso-semantics-4"
SEMANTIC_ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-,.?+"
PAD_ID = 0
CLS_ID = 1
CHAR_TO_ID = {character: index + 2 for index, character in enumerate(SEMANTIC_ALPHABET)}

ACT_NAMES = (
    "CQ", "DIRECTED", "REPLY", "IDENTIFY", "ASK", "PROVIDE", "REPORT",
    "ACK", "CONFIRM", "CORRECTION", "REPEAT_REQUEST", "SPEED_REQUEST",
    "GREETING", "THANKS", "HANDOVER", "SIGNOFF",
)
TOPIC_NAMES = (
    "CALLSIGN", "NAME", "AGE", "LOCATION", "WEATHER",
    "POWER", "RIG", "ANTENNA", "RST", "REGION", "CQ_SCOPE",
)
SLOT_NAMES = ("O",) + TOPIC_NAMES
PROCEDURE_GRADES = (
    "CANONICAL", "INTELLIGIBLE_NONCANONICAL", "INCOMPLETE", "AMBIGUOUS", "IRRELEVANT",
)
REGISTER_NAMES = (
    "CW_PROCEDURAL", "CW_ABBREVIATED", "NATURAL_TEXT", "MIXED", "NOISE_UNKNOWN",
)
PHASE_NAMES = ("IDLE", "CALLING", "WAITING_REPLY", "EXCHANGE", "CLOSING")
PENDING_QUESTION_NAMES = ("NONE",) + TOPIC_NAMES

ACT_TO_ID = {name: index for index, name in enumerate(ACT_NAMES)}
TOPIC_TO_ID = {name: index for index, name in enumerate(TOPIC_NAMES)}
SLOT_TO_ID = {name: index for index, name in enumerate(SLOT_NAMES)}
PROCEDURE_TO_ID = {name: index for index, name in enumerate(PROCEDURE_GRADES)}
REGISTER_TO_ID = {name: index for index, name in enumerate(REGISTER_NAMES)}
PHASE_TO_ID = {name: index for index, name in enumerate(PHASE_NAMES)}
PENDING_QUESTION_TO_ID = {name: index for index, name in enumerate(PENDING_QUESTION_NAMES)}

# All names and places are fictional game data. They do not map to real callsigns.
FICTIONAL_NAMES = ("AKI", "HANA", "KEN", "MIO", "REN", "SORA", "YUKI", "LEO", "MAYA", "NINA")
FICTIONAL_LOCATIONS = ("AOBA", "LAKE HILL", "PINE BAY", "SORA VALLEY", "WEST RIDGE", "MIZU PORT")
WEATHER_VALUES = ("SUNNY", "CLOUDY", "RAIN", "SNOW", "WINDY", "CLEAR")
RIG_VALUES = ("USDX", "MICA8", "PIXIE", "QRP ONE", "HOME BREW")
ANTENNA_VALUES = ("DIPOLE", "VERTICAL", "LOOP", "YAGI", "END FED")
MAX_SEMANTIC_LENGTH = 128
MAX_CALLSIGN_LENGTH = 8
OPTIONAL_TOPIC_NAMES = (
    "NAME", "AGE", "LOCATION", "WEATHER", "POWER", "RIG", "ANTENNA", "RST",
)


def normalize_semantic_text(value: str) -> str:
    cleaned = "".join(character for character in str(value).upper() if character in CHAR_TO_ID)
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class SemanticContext:
    phase: str = "IDLE"
    pending_question: str = "NONE"
    self_callsign: str = ""
    peer_callsign: str = ""
    known_slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticSpec:
    text: str
    acts: tuple[str, ...]
    topics: tuple[str, ...]
    procedure: str
    spans: tuple[tuple[int, int, str], ...] = ()
    context: SemanticContext = field(default_factory=SemanticContext)
    safe_to_commit: bool = True
    register: str = "CW_PROCEDURAL"


def _callsign(rng: np.random.Generator) -> str:
    prefix = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=int(rng.integers(1, 3))))
    digit = str(rng.integers(0, 10))
    suffix = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=int(rng.integers(1, 4))))
    return f"{prefix}{digit}{suffix}"[:7]


def _known_slots(
    rng: np.random.Generator,
    *,
    exclude: tuple[str, ...] = (),
) -> tuple[str, ...]:
    available = tuple(topic for topic in TOPIC_NAMES if topic not in exclude)
    count = int(rng.integers(0, min(4, len(available)) + 1))
    if count == 0:
        return ()
    return tuple(sorted(str(value) for value in rng.choice(available, size=count, replace=False)))


def _context(
    rng: np.random.Generator,
    *,
    phase: str,
    pending_question: str = "NONE",
    self_callsign: str | None = None,
    peer_callsign: str | None = None,
    known_slots: tuple[str, ...] | None = None,
) -> SemanticContext:
    return SemanticContext(
        phase=phase,
        pending_question=pending_question,
        self_callsign=normalize_semantic_text(self_callsign or _callsign(rng)),
        peer_callsign=normalize_semantic_text(peer_callsign or _callsign(rng)),
        known_slots=known_slots if known_slots is not None else _known_slots(rng),
    )


def _join_tagged(parts: list[tuple[str, str | None]]) -> tuple[str, tuple[tuple[int, int, str], ...]]:
    text = ""
    spans: list[tuple[int, int, str]] = []
    for value, slot in parts:
        value = normalize_semantic_text(value)
        if not value:
            continue
        if text:
            text += " "
        start = len(text)
        text += value
        if slot:
            spans.append((start, len(text), slot))
    return text, tuple(spans)


def _value_for_topic(rng: np.random.Generator, topic: str) -> str:
    if topic == "NAME":
        return str(rng.choice(FICTIONAL_NAMES))
    if topic == "AGE":
        return str(rng.integers(8, 86))
    if topic == "LOCATION":
        return str(rng.choice(FICTIONAL_LOCATIONS))
    if topic == "WEATHER":
        return str(rng.choice(WEATHER_VALUES))
    if topic == "POWER":
        return f"{rng.choice((1, 2, 5, 10, 20, 50, 100))}W"
    if topic == "RIG":
        return str(rng.choice(RIG_VALUES))
    if topic == "ANTENNA":
        return str(rng.choice(ANTENNA_VALUES))
    if topic == "RST":
        return str(rng.choice((339, 449, 559, 579, 589, 599)))
    if topic == "REGION":
        return str(rng.choice(("JA", "EU", "NA", "SA", "AF", "OC")))
    if topic == "CQ_SCOPE":
        return str(rng.choice(("DX", "TEST", "QRP")))
    if topic == "CALLSIGN":
        return _callsign(rng)
    raise ValueError(topic)


QUESTION_PREFIXES = {
    "NAME": ("UR NAME?", "NAME?", "PSE NAME?"),
    "AGE": ("UR AGE?", "AGE?", "HW OLD?"),
    "LOCATION": ("UR QTH?", "QTH?", "PSE QTH?"),
    "WEATHER": ("HW WX?", "WX?", "UR WX?"),
    "POWER": ("UR PWR?", "PWR?", "HW PWR?"),
    "RIG": ("UR RIG?", "RIG?", "PSE RIG?"),
    "ANTENNA": ("UR ANT?", "ANT?", "PSE ANT?"),
    "RST": ("RST?", "HW CPY?", "PSE RST?"),
    "CALLSIGN": ("UR CALL?", "CALL?", "AGN CALL?"),
}

NATURAL_QUESTIONS = {
    "NAME": "WHAT IS YOUR NAME?", "AGE": "HOW OLD ARE YOU?",
    "LOCATION": "WHERE ARE YOU LOCATED?", "WEATHER": "HOW IS THE WEATHER?",
    "POWER": "WHAT POWER DO YOU USE?", "RIG": "WHAT RADIO DO YOU USE?",
    "ANTENNA": "WHAT ANTENNA DO YOU USE?", "RST": "WHAT IS MY SIGNAL REPORT?",
}

MIXED_QUESTIONS = {
    "NAME": "PSE TELL ME UR NAME?", "AGE": "PSE TELL ME UR AGE?",
    "LOCATION": "PSE TELL ME UR QTH?", "WEATHER": "PSE TELL ME UR WX?",
    "POWER": "PSE TELL ME UR PWR?", "RIG": "PSE TELL ME UR RIG?",
    "ANTENNA": "PSE TELL ME UR ANT?", "RST": "PSE GIVE ME RST?",
}

INCOMPLETE_QUESTIONS = {
    "NAME": "UR NAME", "AGE": "UR AGE", "LOCATION": "UR QTH", "WEATHER": "UR WX",
    "POWER": "UR PWR", "RIG": "UR RIG", "ANTENNA": "UR ANT", "RST": "UR RST",
}

ANSWER_KEYS = {
    "NAME": "NAME", "AGE": "AGE", "LOCATION": "QTH", "WEATHER": "WX",
    "POWER": "PWR", "RIG": "RIG", "ANTENNA": "ANT", "RST": "RST", "CALLSIGN": "CALL",
}


def _question_text(rng: np.random.Generator, topics: tuple[str, ...], style: str) -> str:
    if style == "CANONICAL":
        values = (str(rng.choice(QUESTION_PREFIXES[topic])) for topic in topics)
    elif style == "NATURAL":
        values = (NATURAL_QUESTIONS[topic] for topic in topics)
    elif style == "MIXED":
        values = (MIXED_QUESTIONS[topic] for topic in topics)
    elif style == "INCOMPLETE":
        values = (INCOMPLETE_QUESTIONS[topic] for topic in topics)
    else:
        raise ValueError(style)
    return normalize_semantic_text(" ".join(values))


def _answer_parts(topic: str, value: str, style: str) -> list[tuple[str, str | None]]:
    if style == "CANONICAL":
        return [(ANSWER_KEYS[topic], None), (value, topic)]
    if style == "NATURAL":
        templates = {
            "NAME": [("MY NAME IS", None), (value, topic)],
            "AGE": [("I AM", None), (value, topic), ("YEARS OLD", None)],
            "LOCATION": [("MY LOCATION IS", None), (value, topic)],
            "WEATHER": [("THE WEATHER IS", None), (value, topic)],
            "POWER": [("I USE", None), (value, topic), ("POWER", None)],
            "RIG": [("MY RADIO IS", None), (value, topic)],
            "ANTENNA": [("MY ANTENNA IS", None), (value, topic)],
            "RST": [("YOUR SIGNAL REPORT IS", None), (value, topic)],
        }
        return templates[topic]
    if style == "MIXED":
        prefixes = {
            "NAME": "NAME IS", "AGE": "AGE IS", "LOCATION": "QTH IS",
            "WEATHER": "WX IS", "POWER": "PWR IS", "RIG": "RIG IS",
            "ANTENNA": "ANT IS", "RST": "UR RST IS",
        }
        return [(prefixes[topic], None), (value, topic)]
    if style == "INCOMPLETE":
        fragments = {
            "NAME": "MY NAME IS", "AGE": "MY AGE IS", "LOCATION": "QTH IS",
            "WEATHER": "WX IS", "POWER": "PWR IS", "RIG": "RIG IS",
            "ANTENNA": "ANT IS", "RST": "UR RST IS",
        }
        return [(fragments[topic], None)]
    raise ValueError(style)


def build_omitted_answer_spec(
    topic: str,
    value: str,
    *,
    pending_question: str = "NONE",
    phase: str = "EXCHANGE",
    self_callsign: str = "SIM1AA",
    peer_callsign: str = "SIM2BB",
    known_slots: tuple[str, ...] = (),
) -> SemanticSpec:
    """Build paired contextual/bare examples without inferring from the value alone."""
    if topic not in TOPIC_TO_ID or topic == "CALLSIGN":
        raise ValueError(topic)
    normalized_value = normalize_semantic_text(value)
    text, spans = _join_tagged([(normalized_value, topic), ("K", None)])
    context = SemanticContext(
        phase=phase,
        pending_question=pending_question,
        self_callsign=normalize_semantic_text(self_callsign),
        peer_callsign=normalize_semantic_text(peer_callsign),
        known_slots=tuple(sorted(slot for slot in known_slots if slot in TOPIC_TO_ID)),
    )
    is_contextual_answer = phase == "EXCHANGE" and pending_question == topic
    if is_contextual_answer:
        return SemanticSpec(
            text, ("PROVIDE",), (topic,), "INTELLIGIBLE_NONCANONICAL", spans,
            context, True, "CW_ABBREVIATED",
        )
    return SemanticSpec(text, (), (), "AMBIGUOUS", (), context, False, "CW_ABBREVIATED")


CONTROL_SPECS = (
    ("AGN? K", ("ASK", "REPEAT_REQUEST", "HANDOVER"), "CANONICAL", True),
    ("PSE AGN K", ("ASK", "REPEAT_REQUEST", "HANDOVER"), "CANONICAL", True),
    ("?", ("ASK", "REPEAT_REQUEST"), "INCOMPLETE", True),
    ("QRS PSE K", ("ASK", "SPEED_REQUEST", "HANDOVER"), "CANONICAL", True),
    ("PSE QRS K", ("ASK", "SPEED_REQUEST", "HANDOVER"), "CANONICAL", True),
    ("QRS? K", ("ASK", "SPEED_REQUEST", "HANDOVER"), "CANONICAL", True),
    ("QRQ K", ("ASK", "SPEED_REQUEST", "HANDOVER"), "CANONICAL", True),
    ("R K", ("ACK", "CONFIRM", "HANDOVER"), "CANONICAL", True),
    ("CFM K", ("CONFIRM", "HANDOVER"), "CANONICAL", True),
    ("HH CORR K", ("CORRECTION", "HANDOVER"), "CANONICAL", True),
    ("GM OM K", ("GREETING", "HANDOVER"), "CANONICAL", True),
    ("GA DR OM K", ("GREETING", "HANDOVER"), "CANONICAL", True),
    ("GE OM TNX K", ("GREETING", "THANKS", "ACK", "HANDOVER"), "CANONICAL", True),
)

HARD_NEGATIVE_SPECS = (
    ("CQ DE K", ("CQ",), ()),
    ("RST K", ("PROVIDE", "REPORT"), ("RST",)),
    ("PWR K", ("PROVIDE",), ("POWER",)),
    ("AGN CQ", ("ASK", "REPEAT_REQUEST", "CQ"), ()),
    ("DE 73 K", ("SIGNOFF", "HANDOVER"), ()),
)

STYLE_REGISTERS = {
    "CANONICAL": "CW_ABBREVIATED",
    "NATURAL": "NATURAL_TEXT",
    "MIXED": "MIXED",
    "INCOMPLETE": "CW_ABBREVIATED",
}


def generate_semantic_spec(rng: np.random.Generator) -> SemanticSpec:
    mode = str(rng.choice(
        (
            "CQ", "STANDARD", "CONTROL", "QUESTION", "ANSWER", "OMITTED_VALID",
            "OMITTED_AMBIGUOUS", "HARD_NEGATIVE", "IRRELEVANT",
        ),
        p=(0.16, 0.15, 0.12, 0.16, 0.19, 0.08, 0.07, 0.04, 0.03),
    ))
    if mode == "CQ":
        self_call = _callsign(rng)
        target = str(rng.choice(("", "", "DX", "JA", "EU", "NA", "QRP", "TEST")))
        target_topic = "REGION" if target in ("JA", "EU", "NA") else "CQ_SCOPE"
        parts = [("CQ", None)] * int(rng.integers(1, 4))
        if target:
            parts.append((target, target_topic))
        parts.extend((("DE", None), (self_call, "CALLSIGN")))
        if rng.random() < 0.25:
            parts.append((self_call, "CALLSIGN"))
        if rng.random() < 0.2:
            parts.append(("PSE", None))
        parts.append((str(rng.choice(("K", "AR"))), None))
        text, spans = _join_tagged(parts)
        acts = ("CQ", "IDENTIFY", "HANDOVER") + (("DIRECTED",) if target else ())
        topics = ("CALLSIGN",) + ((target_topic,) if target else ())
        context = _context(rng, phase="CALLING", self_callsign=self_call)
        return SemanticSpec(text, acts, topics, "CANONICAL", spans, context, True, "CW_PROCEDURAL")

    if mode == "STANDARD":
        peer_call = _callsign(rng)
        self_call = _callsign(rng)
        rst = _value_for_topic(rng, "RST")
        parts = [
            (peer_call, "CALLSIGN"), ("DE", None), (self_call, "CALLSIGN"),
            ("UR RST", None), (rst, "RST"), ("TNX 73 K", None),
        ]
        text, spans = _join_tagged(parts)
        context = _context(rng, phase="EXCHANGE", self_callsign=self_call, peer_callsign=peer_call)
        acts = ("REPLY", "IDENTIFY", "REPORT", "ACK", "THANKS", "HANDOVER", "SIGNOFF")
        return SemanticSpec(text, acts, ("CALLSIGN", "RST"), "CANONICAL", spans, context)

    if mode == "CONTROL":
        text, acts, procedure, safe = CONTROL_SPECS[int(rng.integers(0, len(CONTROL_SPECS)))]
        context = _context(rng, phase="EXCHANGE")
        return SemanticSpec(text, acts, (), procedure, (), context, safe, "CW_ABBREVIATED")

    if mode == "QUESTION":
        count = 1 if rng.random() < 0.82 else 2
        topics = tuple(str(value) for value in rng.choice(OPTIONAL_TOPIC_NAMES, size=count, replace=False))
        style = str(rng.choice(("CANONICAL", "NATURAL", "MIXED", "INCOMPLETE"), p=(0.55, 0.18, 0.17, 0.10)))
        procedure = {
            "CANONICAL": "CANONICAL", "NATURAL": "INTELLIGIBLE_NONCANONICAL",
            "MIXED": "INTELLIGIBLE_NONCANONICAL", "INCOMPLETE": "INCOMPLETE",
        }[style]
        context = _context(rng, phase="EXCHANGE", known_slots=_known_slots(rng, exclude=topics))
        acts = ("ASK", "HANDOVER") if style != "INCOMPLETE" else ("ASK",)
        text = _question_text(rng, topics, style)
        if style != "INCOMPLETE":
            text = f"{text} K"
        return SemanticSpec(
            text, acts, topics, procedure, (), context, procedure != "INCOMPLETE",
            STYLE_REGISTERS[style],
        )

    if mode == "ANSWER":
        count = int(rng.choice((1, 2, 3), p=(0.60, 0.30, 0.10)))
        topics = tuple(str(value) for value in rng.choice(OPTIONAL_TOPIC_NAMES, size=count, replace=False))
        style = str(rng.choice(("CANONICAL", "NATURAL", "MIXED", "INCOMPLETE"), p=(0.55, 0.18, 0.17, 0.10)))
        parts: list[tuple[str, str | None]] = []
        if style != "INCOMPLETE" and rng.random() < 0.25:
            parts.append(("TNX", None))
        for topic in topics:
            parts.extend(_answer_parts(topic, _value_for_topic(rng, topic), style))
        if style != "INCOMPLETE" and rng.random() < 0.25:
            parts.append(("HW?", None))
        if style != "INCOMPLETE":
            parts.append(("K", None))
        text, spans = _join_tagged(parts)
        acts = ("PROVIDE", "ACK") if text.startswith("TNX") else ("PROVIDE",)
        if "HW?" in text:
            acts += ("ASK",)
        if style != "INCOMPLETE":
            acts += ("HANDOVER",)
        procedure = {
            "CANONICAL": "CANONICAL", "NATURAL": "INTELLIGIBLE_NONCANONICAL",
            "MIXED": "INTELLIGIBLE_NONCANONICAL", "INCOMPLETE": "INCOMPLETE",
        }[style]
        pending = topics[0] if rng.random() < 0.55 else "NONE"
        context = _context(
            rng, phase="EXCHANGE", pending_question=pending,
            known_slots=_known_slots(rng, exclude=topics),
        )
        return SemanticSpec(
            text, acts, topics, procedure, spans, context, procedure != "INCOMPLETE",
            STYLE_REGISTERS[style],
        )

    if mode in ("OMITTED_VALID", "OMITTED_AMBIGUOUS"):
        topic = str(rng.choice(OPTIONAL_TOPIC_NAMES))
        value = _value_for_topic(rng, topic)
        self_call, peer_call = _callsign(rng), _callsign(rng)
        known = _known_slots(rng, exclude=(topic,))
        if mode == "OMITTED_VALID":
            pending, phase = topic, "EXCHANGE"
        else:
            ambiguity = int(rng.integers(0, 3))
            if ambiguity == 0:
                pending, phase = "NONE", "EXCHANGE"
            elif ambiguity == 1:
                pending = str(rng.choice(tuple(name for name in OPTIONAL_TOPIC_NAMES if name != topic)))
                phase = "EXCHANGE"
            else:
                pending, phase = topic, str(rng.choice(("IDLE", "CALLING", "CLOSING")))
        return build_omitted_answer_spec(
            topic, value, pending_question=pending, phase=phase,
            self_callsign=self_call, peer_callsign=peer_call, known_slots=known,
        )

    if mode == "HARD_NEGATIVE":
        text, acts, topics = HARD_NEGATIVE_SPECS[int(rng.integers(0, len(HARD_NEGATIVE_SPECS)))]
        context = _context(rng, phase=str(rng.choice(("CALLING", "EXCHANGE", "CLOSING"))))
        return SemanticSpec(
            text, acts, topics, "INCOMPLETE", (), context, False, "CW_ABBREVIATED",
        )

    context = _context(rng, phase=str(rng.choice(PHASE_NAMES)))
    noise = str(rng.choice(("T T T T", "12345", "QXZ QXZ", "...", "VVV VVV")))
    return SemanticSpec(noise, (), (), "IRRELEVANT", (), context, False, "NOISE_UNKNOWN")


def _encode_callsign(value: str) -> torch.Tensor:
    encoded = torch.zeros(MAX_CALLSIGN_LENGTH, dtype=torch.long)
    normalized = normalize_semantic_text(value).replace(" ", "")[:MAX_CALLSIGN_LENGTH]
    for index, character in enumerate(normalized):
        encoded[index] = CHAR_TO_ID[character]
    return encoded


def encode_semantic_spec(spec: SemanticSpec) -> dict[str, Any]:
    text = normalize_semantic_text(spec.text)[: MAX_SEMANTIC_LENGTH - 1]
    input_ids = torch.zeros(MAX_SEMANTIC_LENGTH, dtype=torch.long)
    valid_mask = torch.zeros(MAX_SEMANTIC_LENGTH, dtype=torch.bool)
    slot_labels = torch.full((MAX_SEMANTIC_LENGTH,), -100, dtype=torch.long)
    input_ids[0] = CLS_ID
    valid_mask[0] = True
    slot_labels[0] = SLOT_TO_ID["O"]
    for index, character in enumerate(text, start=1):
        input_ids[index] = CHAR_TO_ID[character]
        valid_mask[index] = True
        slot_labels[index] = SLOT_TO_ID["O"]
    for start, end, slot in spec.spans:
        clipped_end = min(end, len(text))
        if start < clipped_end:
            slot_labels[start + 1:clipped_end + 1] = SLOT_TO_ID[slot]
    acts = torch.zeros(len(ACT_NAMES), dtype=torch.float32)
    topics = torch.zeros(len(TOPIC_NAMES), dtype=torch.float32)
    known_slots = torch.zeros(len(TOPIC_NAMES), dtype=torch.float32)
    for name in spec.acts:
        acts[ACT_TO_ID[name]] = 1.0
    for name in spec.topics:
        topics[TOPIC_TO_ID[name]] = 1.0
    for name in spec.context.known_slots:
        if name in TOPIC_TO_ID:
            known_slots[TOPIC_TO_ID[name]] = 1.0
    return {
        "input_ids": input_ids,
        "valid_mask": valid_mask,
        "context_ids": torch.tensor((
            PHASE_TO_ID.get(spec.context.phase, PHASE_TO_ID["IDLE"]),
            PENDING_QUESTION_TO_ID.get(spec.context.pending_question, PENDING_QUESTION_TO_ID["NONE"]),
        ), dtype=torch.long),
        "self_callsign_ids": _encode_callsign(spec.context.self_callsign),
        "peer_callsign_ids": _encode_callsign(spec.context.peer_callsign),
        "known_slots": known_slots,
        "slot_labels": slot_labels,
        "acts": acts,
        "topics": topics,
        "register": torch.tensor(REGISTER_TO_ID[spec.register], dtype=torch.long),
        "procedure": torch.tensor(PROCEDURE_TO_ID[spec.procedure], dtype=torch.long),
        "safe_to_commit": torch.tensor(float(spec.safe_to_commit), dtype=torch.float32),
        "text": text,
        "spec": spec,
    }


class SyntheticSemanticDataset(Dataset):
    def __init__(self, samples: int, *, seed: int) -> None:
        self.samples = int(samples)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + int(index) * 1_000_003)
        return encode_semantic_spec(generate_semantic_spec(rng))


def collate_semantic(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "input_ids", "valid_mask", "context_ids", "self_callsign_ids", "peer_callsign_ids",
        "known_slots", "slot_labels", "acts", "topics", "register", "procedure", "safe_to_commit",
    )
    return {
        **{key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys},
        "texts": [sample["text"] for sample in samples],
        "contexts": [sample["spec"].context for sample in samples],
    }
