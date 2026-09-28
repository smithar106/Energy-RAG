from app.ingestion.sources.eia_articles import _valid_link


def test_valid_eia_article_link():
    assert _valid_link("https://www.eia.gov/todayinenergy/detail.php?id=68204")


def test_malformed_eia_links_rejected():
    assert not _valid_link("https://www.eia.gov/todayinenergy/detail.php?id=")
    assert not _valid_link("")
    assert not _valid_link("https://www.eia.gov/todayinenergy/")
