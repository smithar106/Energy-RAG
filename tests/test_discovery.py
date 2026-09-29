from datetime import date

from app.retrieval.discovery import _event_years, _title_relevant
from app.retrieval.intent import RetrievalIntent


def _intent(energy="electricity", causal=True):
    return RetrievalIntent(
        energy_type=energy, market_layer="retail_price", geography="United States",
        sector="all", event_start=date(2017, 5, 1), event_end=date(2017, 6, 1),
        is_causal=causal,
    )


def test_title_relevance_electricity():
    intent = _intent()
    assert _title_relevant("Natural gas generators make up the largest share of U.S. generation", intent)
    assert _title_relevant("Residential electricity prices up in first half of 2017", intent)
    assert not _title_relevant("Crude oil production reaches record high", intent)
    assert not _title_relevant("U.S. gasoline exports increase", intent)


def test_event_years_causal_includes_context():
    intent = _intent(causal=True)
    assert _event_years(intent) == [2016, 2017, 2018]
    assert _event_years(_intent(causal=False)) == [2017]


def test_no_event_years():
    intent = RetrievalIntent("electricity", None, None, None, None, None, False)
    assert _event_years(intent) == []
