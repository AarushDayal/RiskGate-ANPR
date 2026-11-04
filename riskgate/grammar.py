import re

STANDARD = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$")
BH = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")


def classify(text: str) -> str:
    """VALID (standard Indian format), NONSTANDARD (plate-like but off-format), INVALID (junk).

    # ponytail: NONSTANDARD/INVALID boundary is a heuristic (>=6 alnum with both letters
    # and digits); if a real licence plate catalogue shows up, replace with lookup.
    """
    t = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if STANDARD.match(t) or BH.match(t):
        return "VALID"
    if len(t) >= 6 and t.isalnum() and any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
        return "NONSTANDARD"
    return "INVALID"
