from datetime import date

from app.retrieval.ranking import authority_for, rank_chunks, temporal_score
from app.retrieval.temporal import TimePeriod


def test_authority_eia_outranks_wikipedia():
    eia = authority_for("U.S. Energy Information Administration")
    wiki = authority_for("Wikipedia")
    assert eia > wiki
    assert authority_for(None) < wiki


def test_temporal_overlap_scores_higher_than_gap():
    overlap, reason = temporal_score(2020, 2024, 2021, 2023)
    gap, gap_reason = temporal_score(2020, 2024, 1990, 1995)
    assert reason == "overlap"
    assert gap_reason == "gap"
    assert overlap > gap


def test_temporal_neutral_without_period_or_window():
    assert temporal_score(None, None, 2020, 2022)[0] == 0.5
    assert temporal_score(2020, 2022, None, None)[0] == 0.5


def test_authority_breaks_ties():
    period = TimePeriod(start=date(2021, 1, 1), end=date(2023, 12, 31))
    candidates = [
        {"id": 1, "similarity": 0.80, "source_name": "Wikipedia",
         "start_year": 2021, "end_year": 2023, "text": "x"},
        {"id": 2, "similarity": 0.80, "source_name": "U.S. Energy Information Administration",
         "start_year": 2021, "end_year": 2023, "text": "y"},
    ]
    ranked = rank_chunks(candidates, period=period, top_n=2)
    assert ranked[0]["id"] == 2
    assert ranked[0]["final_score"] >= ranked[1]["final_score"]


def test_temporal_boost_can_outrank_semantics():
    period = TimePeriod(start=date(2021, 1, 1), end=date(2023, 12, 31))
    candidates = [
        {"id": 1, "similarity": 0.80, "source_name": "Wikipedia",
         "start_year": 1990, "end_year": 1995, "text": "x"},
        {"id": 2, "similarity": 0.70, "source_name": "Wikipedia",
         "start_year": 2021, "end_year": 2023, "text": "y"},
    ]
    ranked = rank_chunks(candidates, period=period, top_n=2)
    assert ranked[0]["id"] == 2
