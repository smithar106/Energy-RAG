from app.ingestion.cleaner import clean_text


def test_strips_html():
    assert clean_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_collapses_whitespace():
    assert clean_text("a\n\n  b\t c") == "a b c"


def test_removes_citation_boilerplate():
    assert clean_text("Prices rose[1] in 2022[edit].") == "Prices rose in 2022."
