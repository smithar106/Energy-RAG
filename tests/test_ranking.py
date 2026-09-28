from datetime import date

from app.retrieval.ranking import (
    authority_for,
    rank_chunks,
    select_evidence,
    temporal_score,
)
from app.retrieval.temporal import TimePeriod


def test_authority_order():
    assert authority_for("U.S. Energy Information Administration") == 1.0
    assert authority_for("U.S. Department of Energy") == 0.9
    assert authority_for("Some Institute") == 0.6
    assert authority_for("Wikipedia") == 0.4


def test_temporal_overlap_beats_gap():
    overlap, reason, _ = temporal_score(2020, 2024, 2021, 2023)
    gap, gap_reason, _ = temporal_score(2020, 2024, 1990, 1995)
    assert reason == "overlap"
    assert gap_reason == "gap"
    assert overlap > gap


def test_temporal_specificity_narrow_beats_broad():
    exact, _, _ = temporal_score(2017, 2017, 2017, 2017)
    narrow, _, _ = temporal_score(2017, 2017, 2016, 2018)
    broad, _, _ = temporal_score(2017, 2017, 2007, 2020)
    assert exact > narrow > broad
    # A very broad window must not dominate merely by containing the year.
    assert broad < 0.6


def test_temporal_neutral_without_period_or_window():
    assert temporal_score(None, None, 2020, 2022)[0] == 0.5
    assert temporal_score(2020, 2022, None, None)[0] == 0.5


def test_diversity_caps_chunks_per_document():
    period = TimePeriod(start=date(2017, 1, 1), end=date(2017, 12, 31))
    candidates = [
        {"id": i, "similarity": 0.85, "source_name": "U.S. Energy Information Administration",
         "start_year": 2017, "end_year": 2017, "document_id": 1, "text": "x"} for i in range(6)
    ] + [
        {"id": i, "similarity": 0.72, "source_name": "U.S. Energy Information Administration",
         "start_year": 2017, "end_year": 2017, "document_id": 2, "text": "y"} for i in range(6, 8)
    ]
    ranked = rank_chunks(candidates, period=period)
    accepted, _ = select_evidence(ranked, top_n=5, max_per_doc=3)
    docs = [c["document_id"] for c in accepted]
    assert 2 in docs                       # a second document is included
    assert docs.count(1) <= 3              # no document monopolises the slots


def test_threshold_rejects_weak_evidence():
    period = TimePeriod(start=date(2017, 1, 1), end=date(2017, 12, 31))
    candidates = [{
        "id": 0, "similarity": 0.05, "source_name": "Wikipedia",
        "start_year": 1990, "end_year": 1991, "document_id": 1, "text": "x",
    }]
    ranked = rank_chunks(candidates, period=period)
    accepted, rejected = select_evidence(ranked, top_n=5)
    assert accepted == []
    assert len(rejected) == 1
