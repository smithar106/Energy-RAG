from app.ingestion.html import clean_inline, html_to_segments


def test_clean_inline_strips_citations():
    assert clean_inline("Prices rose[1] in 2022.[citation needed]") == "Prices rose in 2022."


def test_html_to_segments_splits_on_headings():
    html = (
        "<h2>Alpha</h2><p>First para.</p><p>Second para.</p>"
        "<h2>Beta</h2><p>Beta para.</p>"
    )
    segments = html_to_segments(html)
    headings = [s.heading for s in segments]
    assert "Alpha" in headings and "Beta" in headings
    alpha = next(s for s in segments if s.heading == "Alpha")
    assert "First para." in alpha.text
    assert "Second para." in alpha.text


def test_boilerplate_sections_dropped():
    html = "<h2>History</h2><p>Real content here.</p><h2>References</h2><p>Ref one.</p>"
    segments = html_to_segments(html)
    assert all(s.heading != "References" for s in segments)
    assert any(s.heading == "History" for s in segments)
