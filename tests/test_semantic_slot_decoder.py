from cw_pulseformer.semantic_slot_decoder import (
    decode_critical_slot_values,
    labeled_slot_values,
)


def test_critical_decoder_rejects_the_w_in_the_word_power() -> None:
    decoded = decode_critical_slot_values(
        "TNX WX SUNNY RST 589 PWR 5W POWER K",
        {"RST": 0.99, "POWER": 0.99},
        safe_to_commit=True,
    )
    assert decoded == {"RST": ("589",), "POWER": ("5W",)}


def test_critical_decoder_is_fail_closed() -> None:
    assert decode_critical_slot_values(
        "100W K", {"POWER": 1.0}, safe_to_commit=False,
    ) == {}


def test_critical_decoder_deduplicates_repeated_callsigns() -> None:
    decoded = decode_critical_slot_values(
        "CQ DE SIM1AA SIM1AA K", {"CALLSIGN": 1.0}, safe_to_commit=True,
    )
    assert decoded["CALLSIGN"] == ("SIM1AA",)


def test_labeled_values_keep_distinct_contiguous_spans() -> None:
    text = "5W POWER"
    labels = [0, 6, 6, 0, 0, 0, 0, 0, 0]
    assert labeled_slot_values(text, labels, 6) == ("5W",)
