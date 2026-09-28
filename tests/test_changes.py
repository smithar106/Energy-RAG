from datetime import date

from app.retrieval.changes import _build_change


def test_absolute_and_percent_from_same_pair():
    change = _build_change(
        calculation_type="month_over_month",
        series_id="ELEC.PRICE.US-ALL.M",
        units="cents/kWh",
        prev_period=date(2017, 5, 1), prev_value=10.34,
        curr_period=date(2017, 6, 1), curr_value=10.83,
    )
    assert round(change.absolute_change, 2) == 0.49
    assert round(change.percent_change, 2) == 4.74
    assert change.previous_period == "2017-05-01"
    assert change.current_period == "2017-06-01"
    assert change.energy_type == "electricity"
    assert change.unit == "cents/kWh"


def test_negative_and_small_changes():
    change = _build_change(
        calculation_type="month_over_month",
        series_id="X", units="u",
        prev_period=date(2017, 6, 1), prev_value=10.83,
        curr_period=date(2017, 7, 1), curr_value=10.95,
    )
    assert round(change.absolute_change, 2) == 0.12
    assert round(change.percent_change, 2) == 1.11


def test_zero_previous_yields_no_percent():
    change = _build_change(
        calculation_type="month_over_month",
        series_id="X", units="u",
        prev_period=date(2017, 1, 1), prev_value=0.0,
        curr_period=date(2017, 2, 1), curr_value=1.0,
    )
    assert change.percent_change is None
