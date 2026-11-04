from riskgate.grammar import classify

# 10 valid (hand-checked against the two formats)
VALID = [
    "KL34A4651",
    "MH12DE1234",
    "DL3CD1210",
    "RJ14AB1234",
    "KA01AJ7533",
    "TN58D5353",
    "MP07L7524",
    "KL35F4337",
    "22BH1234A",
    "22BH1234AB",
]

# 10 invalid (not plate-like: empty, too short, digits only, letters only, garbage)
INVALID = ["", "  ", "1234", "12345", "A", "AB", "1A", "ABCDEFGH", "@#$%", "246802"]


def test_valid():
    for s in VALID:
        assert classify(s) == "VALID", s


def test_nonstandard():
    # plate-like (>=6 alnum, both letters+digits) but off-format
    assert classify("KL41L7OOT") == "NONSTANDARD"
    assert classify("INDKL07BX7197") == "NONSTANDARD"
    assert classify("409C2768") == "NONSTANDARD"


def test_invalid():
    for s in INVALID:
        assert classify(s) == "INVALID", s


def test_no_correction():
    # classify must never "fix" text - close-but-wrong stays NONSTANDARD, not VALID
    assert classify("KL34A46") != "VALID"
