from datetime import date

from app.ingestion.domain import classify_domain
from app.ingestion.metadata import extract_event_window, extract_mentioned_years
from app.retrieval.intent import parse_intent
from app.retrieval.query_builder import build_retrieval_queries


def test_classify_gasoline_vs_electricity():
    assert classify_domain("gasoline formulation and motor fuel prices at the pump").energy_type == "gasoline"
    assert classify_domain("U.S. retail electricity prices rose for residential customers").energy_type == "electricity"
    assert classify_domain("crude oil and petroleum production").energy_type == "petroleum"


def test_event_window_month_range():
    start, end = extract_event_window(
        "In this analysis", title="U.S. electricity prices May-June 2017"
    )
    assert start == date(2017, 5, 1)
    assert end == date(2017, 6, 30)


def test_mentioned_years():
    years = extract_mentioned_years("prices in 2017 and 2018, and again in 2017")
    assert years == [2017, 2018]


def test_intent_from_sql_change():
    sql = [{
        "changes": [{
            "series_id": "ELEC.PRICE.US-ALL.M",
            "previous_period": "2017-05-01", "current_period": "2017-06-01",
            "previous_value": 10.34, "current_value": 10.83,
        }]
    }]
    intent = parse_intent("What caused the biggest increase in US electricity prices in 2017?", sql)
    assert intent.energy_type == "electricity"
    assert intent.market_layer == "retail_price"
    assert intent.geography == "United States"
    assert intent.event_start == date(2017, 5, 1)
    assert intent.event_end == date(2017, 6, 1)
    assert intent.is_causal is True


def test_query_expansion_multiple_and_data_aware():
    sql = [{
        "changes": [{
            "series_id": "ELEC.PRICE.US-ALL.M",
            "previous_period": "2017-05-01", "current_period": "2017-06-01",
        }]
    }]
    intent = parse_intent("What caused the biggest increase in US electricity prices in 2017?", sql)
    queries = build_retrieval_queries("US electricity price increase 2017", intent)
    assert len(queries) >= 4
    assert any("2017" in q for q in queries)
    assert any("generation" in q for q in queries)


def test_event_drives_queries_not_january():
    sql = [{
        "changes": [{
            "series_id": "ELEC.PRICE.US-ALL.M",
            "previous_period": "2017-05-01", "current_period": "2017-06-01",
        }]
    }]
    intent = parse_intent("What caused the biggest increase in US electricity prices in 2017?", sql)
    assert intent.event_start == date(2017, 5, 1)
    assert intent.event_end == date(2017, 6, 1)
    # ±3 month context window, distinct from the exact event.
    assert intent.context_start == date(2017, 2, 1)
    assert intent.context_end == date(2017, 9, 30)
    joined = " ".join(build_retrieval_queries("US electricity price increase", intent)).lower()
    assert "may" in joined and "june" in joined
    assert "january" not in joined
