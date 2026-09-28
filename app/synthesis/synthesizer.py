"""Grounded final-answer generation.

DeepSeek synthesizes the answer, but it is handed *only*:
  - verified SQL numbers (from the structured path), and
  - retrieved real-source evidence chunks (from the vector path).

It is told to cite evidence as ``[n]`` and never to introduce a number, source,
quote, title, URL, or date that is not present in the provided context.
"""
from __future__ import annotations

import json

from app.providers.llm import ChatMessage, get_llm_provider

SYNTHESIS_SYSTEM = """You are a rigorous energy analyst synthesizing a final answer.

You are given:
1. VERIFIED_SQL — verified price data, aggregates, and price-change records.
   These are the ONLY allowed numbers.
2. EVIDENCE — ranked real-source passages, each with an index [n].

Rules:
- State numbers exactly as they appear in VERIFIED_SQL. Do NOT round, invent, combine, or recompute them.
- A price-change record contains the exact observation pair. Report its absolute and percent change together, from that same pair. NEVER mix figures from different pairs and NEVER compute your own percentage.
- If the question asks for the biggest/largest change, LEAD with that change:
  previous period + value, current period + value, absolute change, percent change.
  Do NOT dump the full monthly series unless the user explicitly asks for it.
- Support every historical/causal claim with a citation to EVIDENCE using its index, e.g. [1].
- Never cite a number to EVIDENCE; numbers are backed by SQL only.
- Never invent source titles, URLs, publication dates, or quotes. Use evidence indices only.
- If EVIDENCE is empty or does not support the explanation, say explicitly that you
  could not find sufficient historical evidence. Do not answer from memory.
- Do not use any outside knowledge. Be concise and factual. Prefer EIA evidence.
"""


class Synthesizer:
    def __init__(self) -> None:
        self.llm = get_llm_provider()

    def synthesize(
        self,
        question: str,
        sql_results: list[dict],
        evidence: list[dict],
    ) -> str:
        sql_payload = json.dumps(sql_results, default=str, indent=2)
        evidence_payload = json.dumps(
            [
                {
                    "index": i,
                    "source_name": c.get("source_name"),
                    "source_type": c.get("source_type"),
                    "title": c.get("document_title"),
                    "section": c.get("section"),
                    "published_date": c.get("published_date"),
                    "start_year": c.get("start_year"),
                    "end_year": c.get("end_year"),
                    "text": c.get("text"),
                }
                for i, c in enumerate(evidence)
            ],
            default=str,
            indent=2,
        )

        user = (
            f"QUESTION:\n{question}\n\n"
            f"VERIFIED_SQL:\n{sql_payload}\n\n"
            f"EVIDENCE:\n{evidence_payload}\n\n"
            "Write the final cited answer. Cite evidence with [n]."
        )

        messages = [
            ChatMessage(role="system", content=SYNTHESIS_SYSTEM),
            ChatMessage(role="user", content=user),
        ]
        response = self.llm.chat(messages, temperature=0.0)
        return response.get("content", "")
