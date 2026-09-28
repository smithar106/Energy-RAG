from datetime import date

from app.retrieval.temporal import TimePeriod, parse_time_period


def test_no_year():
    p = parse_time_period("why did energy prices change")
    assert not p.is_bounded


def test_year_only():
    p = parse_time_period("what was the price in 2022")
    assert p.start == date(2022, 1, 1)
    assert p.end == date(2022, 12, 31)


def test_year_and_month():
    p = parse_time_period("gas prices in March 2023")
    assert p.start == date(2023, 3, 1)
    assert p.end == date(2023, 3, 31)


def test_overlap():
    p = TimePeriod(start=date(2022, 1, 1), end=date(2022, 12, 31))
    assert p.overlaps(date(2022, 6, 1), date(2022, 6, 30))
    assert not p.overlaps(date(2021, 1, 1), date(2021, 12, 31))
    # unbounded chunk metadata still passes (can't filter)
    assert p.overlaps(None, None)


def test_between_range():
    p = parse_time_period("Why did electricity prices rise between 2021 and 2023?")
    assert p.start == date(2021, 1, 1)
    assert p.end == date(2023, 12, 31)


def test_hyphen_range():
    p = parse_time_period("energy prices 2014-2016")
    assert p.start == date(2014, 1, 1)
    assert p.end == date(2016, 12, 31)


def test_event_alias_covid():
    p = parse_time_period("What factors affected energy prices during COVID-19?")
    assert p.start == date(2020, 1, 1)
    assert p.end == date(2021, 12, 31)

