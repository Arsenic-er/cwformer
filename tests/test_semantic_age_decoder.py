from cw_pulseformer.semantic_slot_decoder import decode_slot_values


def test_age_decoder_ignores_rst_and_power_numbers() -> None:
    decoded = decode_slot_values(
        "RST 599 I AM 17 YEARS OLD PWR 5W K",
        {"AGE": 1.0, "RST": 1.0, "POWER": 1.0},
        safe_to_commit=True,
    )
    assert decoded["AGE"] == ("17",)
    assert decoded["RST"] == ("599",)
    assert decoded["POWER"] == ("5W",)


def test_age_decoder_handles_abbreviated_and_elliptical_answers() -> None:
    abbreviated = decode_slot_values(
        "AGE IS 84 K", {"AGE": 1.0}, safe_to_commit=True,
    )
    elliptical = decode_slot_values(
        "9 K", {"AGE": 1.0}, safe_to_commit=True,
    )
    assert abbreviated["AGE"] == ("84",)
    assert elliptical["AGE"] == ("9",)
