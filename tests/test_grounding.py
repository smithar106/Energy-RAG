from app.synthesis.grounding import validate_grounding


def _sql_2017() -> list[dict]:
    """SQL result containing BOTH a period (Jan→Dec) and a monthly (May→Jun) change."""
    return [{
        "operation": "period_end_to_end",
        "price_change": {
            "series_id": "ELEC.PRICE.US-ALL.M",
            "previous_period": "2017-01-01", "previous_value": 10.13,
            "current_period": "2017-12-01", "current_value": 10.17,
            "absolute_change": 0.04, "percent_change": 0.39,
        },
        "changes": [{
            "series_id": "ELEC.PRICE.US-ALL.M",
            "previous_period": "2017-05-01", "previous_value": 10.34,
            "current_period": "2017-06-01", "current_value": 10.83,
            "absolute_change": 0.49, "percent_change": 4.74,
        }],
        "derived": {"count": 12, "average": 10.55, "minimum": 10.07, "maximum": 10.95},
        "rows": [
            {"period": "2017-05-01", "price": 10.34, "units": "cents/kWh", "series_id": "ELEC.PRICE.US-ALL.M"},
            {"period": "2017-06-01", "price": 10.83, "units": "cents/kWh", "series_id": "ELEC.PRICE.US-ALL.M"},
        ],
    }]


def test_2017_percentage_from_same_pair_passes():
    answer = (
        "The largest monthly increase was May 2017 (10.34 cents/kWh) to "
        "June 2017 (10.83 cents/kWh): +0.49 cents/kWh, +4.74%."
    )
    check = validate_grounding(answer, _sql_2017(), [], llm_check=False)
    assert check.valid is True
    assert not any(u.startswith("percent:") for u in check.ungrounded)


def test_2017_mixing_pairs_fails():
    # The production bug: May→Jun absolute paired with the Jan→Dec percent.
    answer = (
        "The largest monthly increase was May 2017 (10.34 cents/kWh) to "
        "June 2017 (10.83 cents/kWh): +0.49 cents/kWh, +0.39%."
    )
    check = validate_grounding(answer, _sql_2017(), [], llm_check=False)
    assert check.valid is False
    assert any(u.startswith("percent:") for u in check.ungrounded)


def test_years_are_not_price_claims():
    answer = "In 2017 the average price was 10.55 cents/kWh."
    check = validate_grounding(answer, _sql_2017(), [], llm_check=False)
    assert check.valid is True


def test_all_numbers_grounded():
    sql = [{"derived": {"average": 15.4, "minimum": 12.1, "maximum": 19.2, "count": 36}, "rows": []}]
    answer = "The average price was 15.4 cents, ranging from 12.1 to 19.2."
    check = validate_grounding(answer, sql, [], llm_check=False)
    assert check.valid is True
    assert check.grounded_numeric_claims == check.total_numeric_claims


def test_ungrounded_number_flagged():
    sql = [{"derived": {"average": 15.4}, "rows": []}]
    answer = "The average price was 15.4 cents, but last month it hit 99.9."
    check = validate_grounding(answer, sql, [], llm_check=False)
    assert check.valid is False
    assert any("99.9" in u for u in check.ungrounded)
