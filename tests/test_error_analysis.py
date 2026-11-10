from scripts.error_analysis import category


def test_categories():
    assert category("", "KA01AB1234") == "empty"
    assert category("58APTN5280", "TN58AP5280") == "order_swapped"
    assert category("KAO1AB1234", "KA01AB1234") == "confusable"
    assert category("INDKA01AB1234", "KA01AB1234") == "extra_text"
    assert category("XYZ", "KA01AB1234") == "other"
