from riskgate.metrics import cer, ece, exact_match, normalize, risk_coverage


def test_normalize():
    assert normalize("  kl-34.a 465 ") == "KL34A465"
    assert normalize("b@h!22") == "BH22"
    assert normalize("") == ""


def test_exact_match():
    assert exact_match("Kl34-A465", "KL34A465") is True
    assert exact_match("KL34A465", "KL34A466") is False


def test_cer():
    # hand check: truth KL498262 (8 chars), pred KLA98262 -> 1 substitution -> 1/8
    assert cer("KLA98262", "KL498262") == 0.125
    # lengthening only: truth AB12, pred AB1234 -> 2 inserts -> 0.5
    assert cer("AB1234", "AB12") == 0.5
    # legit >1 case: pred much longer than truth -> distance 6, len(truth)=4 -> 1.5
    assert cer("AB12CD34EF", "AB12") == 1.5
    # empty truth with non-empty pred -> 1.0
    assert cer("ABC", "") == 1.0


def test_ece():
    # hand: bin5 [conf .5 wrong]: 0.5*|0-.5|=.25 ; bin9 [conf .9 right]: 0.5*|1-.9|=.05 -> .30
    assert ece([0.5, 0.9], [False, True]) == 0.3
    # hand: bin5 acc .5 == conf .5, bin9 (conf 1.0) acc 1.0 == conf 1.0 -> 0.0
    assert ece([0.5, 0.5, 1.0, 1.0], [True, False, True, True]) == 0.0


def test_risk_coverage():
    # hand: confs [0.9,0.5,0.8], corrects [T,F,F] -> order .9(ok), .8(ok), .5(wrong)
    assert risk_coverage([0.9, 0.5, 0.8], [True, False, False]) == [
        (1 / 3, 0.0),
        (2 / 3, 0.5),
        (1.0, 2 / 3),
    ]
