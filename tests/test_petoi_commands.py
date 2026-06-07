from robomind.robomind.petoi_commands import GAITS, PETOI_COMMANDS, POSTURES, TRICKS


def test_all_descriptions_non_empty():
    assert all(v.strip() for v in PETOI_COMMANDS.values())


def test_all_keys_non_empty():
    assert all(k.strip() for k in PETOI_COMMANDS)


def test_categories_mutually_exclusive_and_exhaustive():
    all_categorised = set(GAITS) | set(POSTURES) | set(TRICKS)
    assert all_categorised == set(PETOI_COMMANDS)
    assert len(GAITS) + len(POSTURES) + len(TRICKS) == len(PETOI_COMMANDS)


def test_key_commands_present():
    expected = ["wkF", "trF", "sit", "balance", "hi", "pu", "bf", "rest"]
    for key in expected:
        assert key in PETOI_COMMANDS, f"{key!r} missing from PETOI_COMMANDS"


def test_gaits_end_in_f_l_r_or_are_special():
    for k in GAITS:
        assert k[-1] in ("F", "L", "R") or k in ("bk", "hlw"), (
            f"Unexpected gait key: {k!r}"
        )


def test_r_variants_exist_for_every_l_gait():
    l_gaits = [k for k in GAITS if k.endswith("L")]
    for k in l_gaits:
        r_key = k[:-1] + "R"
        assert r_key in PETOI_COMMANDS, (
            f"Missing R-mirror for {k!r}: expected {r_key!r}"
        )


def test_postures_are_stationary():
    for k in POSTURES:
        assert k in PETOI_COMMANDS


def test_no_duplicate_keys():
    keys = list(PETOI_COMMANDS.keys())
    assert len(keys) == len(set(keys))
