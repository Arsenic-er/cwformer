from cw_pulseformer.semantic_slot_decoder import decode_slot_values


def test_catalog_decoder_extracts_optional_game_values() -> None:
    decoded = decode_slot_values(
        "NAME HANA QTH PINE BAY WX SUNNY RIG USDX ANT LOOP AGE 17 K",
        {
            "NAME": 1.0,
            "LOCATION": 1.0,
            "WEATHER": 1.0,
            "RIG": 1.0,
            "ANTENNA": 1.0,
            "AGE": 1.0,
        },
        safe_to_commit=True,
    )
    assert decoded == {
        "NAME": ("HANA",),
        "AGE": ("17",),
        "LOCATION": ("PINE BAY",),
        "WEATHER": ("SUNNY",),
        "RIG": ("USDX",),
        "ANTENNA": ("LOOP",),
    }


def test_catalog_decoder_accepts_save_specific_values() -> None:
    decoded = decode_slot_values(
        "NAME DIEGO QTH LAKE AURORA K",
        {"NAME": 1.0, "LOCATION": 1.0},
        safe_to_commit=True,
        catalogs={"NAME": ("DIEGO",), "LOCATION": ("LAKE AURORA",)},
    )
    assert decoded == {"NAME": ("DIEGO",), "LOCATION": ("LAKE AURORA",)}
