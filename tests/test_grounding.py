from app.synthesis.grounding import validate_grounding


def test_all_numbers_grounded():
    sql = [{"derived": {"average": 15.4, "minimum": 12.1, "maximum": 19.2, "count": 36}, "rows": []}]
    evidence = []
    answer = "The average price was 15.4 cents, ranging from 12.1 to 19.2."
    check = validate_grounding(answer, sql, evidence, llm_check=False)
    assert check.valid is True
    assert check.grounded_numeric_claims == check.total_numeric_claims


def test_ungrounded_number_flagged():
    sql = [{"derived": {"average": 15.4}, "rows": []}]
    evidence = []
    answer = "The average price was 15.4 cents, but last month it hit 99.9."
    check = validate_grounding(answer, sql, evidence, llm_check=False)
    assert check.valid is False
    assert any("99.9" in u for u in check.ungrounded)


def test_years_are_not_price_claims():
    sql = [{"derived": {"average": 15.4}, "rows": []}]
    answer = "In 2022 the average price was 15.4 cents."
    check = validate_grounding(answer, sql, [], llm_check=False)
    assert check.valid is True
