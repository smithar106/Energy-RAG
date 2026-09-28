from datetime import date

from app.retrieval.reranker import rerank
from app.retrieval.temporal import TimePeriod


def test_rerank_orders_by_score():
    period = TimePeriod(start=date(2022, 1, 1), end=date(2022, 12, 31))
    chunks = [
        {"id": 1, "text": "natural gas prices spiked in 2022", "similarity": 0.9,
         "start_date": date(2022, 1, 1), "end_date": date(2022, 12, 31)},
        {"id": 2, "text": "something unrelated to the query", "similarity": 0.4,
         "start_date": date(2010, 1, 1), "end_date": date(2010, 12, 31)},
    ]
    ranked = rerank(chunks, retrieval_query="natural gas prices 2022", period=period)
    assert ranked[0]["id"] == 1


def test_rerank_respects_top_n():
    chunks = [
        {"id": i, "text": f"chunk {i} natural gas", "similarity": 0.5 + i * 0.01,
         "start_date": None, "end_date": None}
        for i in range(10)
    ]
    ranked = rerank(chunks, retrieval_query="natural gas", period=TimePeriod(), top_n=3)
    assert len(ranked) == 3
