from datetime import date

from app.retrieval.intent import RetrievalIntent
from app.retrieval.ranking import (
    authority_for,
    apply_diversity,
    gate_chunks,
    rank_chunks,
    temporal_score,
)


def _intent(energy="electricity", causal=True, start=date(2017, 5, 1), end=date(2017, 6, 1)):
    return RetrievalIntent(
        energy_type=energy, market_layer="retail_price", geography="United States",
        sector="all", event_start=start, event_end=end, is_causal=causal,
    )


def test_authority_order():
    assert authority_for("U.S. Energy Information Administration") == 1.0
    assert authority_for("U.S. Department of Energy") == 0.9
    assert authority_for("Some Institute") == 0.6
    assert authority_for("Wikipedia") == 0.4


def test_temporal_specificity_narrow_beats_broad():
    exact, _, _ = temporal_score(2017, 2017, 2017, 2017)
    narrow, _, _ = temporal_score(2017, 2017, 2016, 2018)
    broad, _, _ = temporal_score(2017, 2017, 2007, 2020)
    assert exact > narrow > broad
    assert broad < 0.6


def test_temporal_neutral_without_period_or_window():
    assert temporal_score(None, None, 2020, 2022)[0] == 0.5
    assert temporal_score(2020, 2022, None, None)[0] == 0.5


def _chunk(id, *, energy="electricity", geo="United States", sy=2017, ey=2017,
           sim=0.7, doc=1, es=None, ee=None, pub=None, text="electricity prices"):
    return {
        "id": id, "document_id": doc, "document_title": "t", "source_name": "U.S. Energy Information Administration",
        "source_url": "u", "text": text, "similarity": sim,
        "energy_type": energy, "market_layer": "retail_price", "geography": geo, "sector": "all",
        "start_year": sy, "end_year": ey, "event_start_date": es, "event_end_date": ee,
        "published_date": pub,
    }


def test_gate_rejects_off_domain_gasoline():
    intent = _intent()
    ranked = rank_chunks([_chunk(1, energy="gasoline", text="gasoline formulation")],
                         intent=intent, query_text="electricity 2017")
    passed, rejected = gate_chunks(ranked, intent=intent)
    assert passed == []
    assert "energy domain mismatch" in rejected[0]["failure_reason"]


def test_gate_rejects_future_event_for_causal_2017():
    intent = _intent()
    ranked = rank_chunks(
        [_chunk(2, es=date(2026, 1, 1), ee=date(2026, 2, 1), pub=date(2026, 2, 5), sy=2026, ey=2026, text="winter storm 2026")],
        intent=intent, query_text="electricity 2017")
    passed, rejected = gate_chunks(ranked, intent=intent)
    assert passed == []
    assert "no temporal relevance" in rejected[0]["failure_reason"]


def test_gate_rejects_timeless_generic_for_causal():
    intent = _intent()
    ranked = rank_chunks(
        [_chunk(3, sy=None, ey=None, es=None, ee=None, pub=None, text="generic electricity infrastructure")],
        intent=intent, query_text="electricity 2017")
    passed, rejected = gate_chunks(ranked, intent=intent)
    assert passed == []
    assert "no temporal relevance" in rejected[0]["failure_reason"]


def test_gate_accepts_relevant_2017_chunk():
    intent = _intent()
    ranked = rank_chunks(
        [_chunk(4, es=date(2017, 5, 1), ee=date(2017, 6, 30), sy=2017, ey=2017, text="natural gas generation costs 2017 electricity")],
        intent=intent, query_text="electricity 2017 generation fuel costs")
    passed, rejected = gate_chunks(ranked, intent=intent)
    assert len(passed) == 1
    assert passed[0]["gate"] == "PASS"


def test_diversity_caps_chunks_per_document():
    passed = [_chunk(i, doc=1) for i in range(5)] + [_chunk(10, doc=2)]
    accepted = apply_diversity(passed, top_n=4, max_per_doc=3)
    docs = [c["document_id"] for c in accepted]
    assert 2 in docs
    assert docs.count(1) <= 3
