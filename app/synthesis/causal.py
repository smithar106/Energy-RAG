"""Causal-usefulness filter (evidence refinement for "why" questions).

After deterministic reranking and gating, DeepSeek classifies which gate-passed
chunks are *causally useful* — i.e. they connect a concrete market factor to the
specific price movement and period. It may only choose among supplied chunks;
it cannot introduce external knowledge.
"""
from __future__ import annotations

import json
import re

from app.providers.llm import ChatMessage, get_llm_provider

_SYSTEM = (
    "You are a strict evidence classifier for causal explanation. Given a question "
    "and a list of evidence chunks, return the indices (as a JSON array of integers) "
    "of the chunks that DIRECTLY and CAUSALLY explain the price movement in the "
    "question. A chunk qualifies ONLY if it identifies a concrete market factor "
    "(fuel cost, generation cost, demand, weather, supply, policy, etc.) that moved "
    "the price in the SAME direction and SAME time period as the question.\n\n"
    "You MUST exclude:\n"
    "- forecasts, projections, or expectations (they do not explain a past movement)\n"
    "- chunks describing the OPPOSITE direction (a decline when asked about a rise)\n"
    "- chunks about a different time period than the question\n"
    "- generic or topically-related context without a specific causal link\n\n"
    "If none causally explain the movement, return []. Respond with ONLY a JSON array."
)


def filter_causally_useful(question: str, evidence: list[dict]) -> list[dict]:
    """Return the subset of evidence chunks that causally support the question."""
    if not evidence:
        return []

    payload = [
        {
            "index": i,
            "source": c.get("source_name"),
            "title": c.get("document_title"),
            "start_year": c.get("start_year"),
            "end_year": c.get("end_year"),
            "text": c.get("text"),
        }
        for i, c in enumerate(evidence)
    ]
    user = "QUESTION:\n" + question + "\n\nEVIDENCE:\n" + json.dumps(payload, default=str)

    llm = get_llm_provider()
    resp = llm.chat(
        [
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user", content=user),
        ],
        temperature=0.0,
    )
    content = resp.get("content", "")
    match = re.search(r"\[[0-9,\s]*\]", content)
    indices: list[int] = []
    if match:
        try:
            indices = [int(i) for i in json.loads(match.group(0))]
        except (ValueError, TypeError):
            indices = []
    return [evidence[i] for i in indices if 0 <= i < len(evidence)]
