from datetime import date

from app.ingestion.metadata import infer_year_range


def test_cluster_around_dominant_year():
    text = (
        "In 2021 prices rose. In 2022 they spiked further. The 2022 crisis was "
        "global. By 2023 prices eased."
    )
    assert infer_year_range(text) == (2021, 2023)


def test_no_years_returns_none():
    assert infer_year_range("no years mentioned here") == (None, None)


def test_publication_date_clamps_future_events():
    start, end = infer_year_range(
        "In 2030 this happened. In 2030 again.", published_date=date(2022, 1, 1)
    )
    assert end <= 2023
    assert start <= end
