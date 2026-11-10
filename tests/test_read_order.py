from riskgate.engines.easyocr_engine import order_boxes


def box(x: int, y: int, text: str):
    return ([[x, y], [x + 10, y], [x + 10, y + 8], [x, y + 8]], text, 0.9)


def test_one_line():
    assert [b[1] for b in order_boxes([box(20, 1, "B"), box(0, 0, "A")])] == ["A", "B"]


def test_two_lines():
    assert [b[1] for b in order_boxes([box(20, 20, "D"), box(20, 0, "B"),
                                       box(0, 21, "C"), box(0, 1, "A")])] == ["A", "B", "C", "D"]
