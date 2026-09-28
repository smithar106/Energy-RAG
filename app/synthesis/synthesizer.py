"""Grounded final-answer generation.

DeepSeek synthesizes the answer, but it is handed *only*:
  - verified SQL numbers (from the structured path), and
  - retrieved evidence chunks (from the vector path).

It is told to cite evidence as [n] and never to introduce a number that is not
present in the provided context.
"""
from __future__ import annotations

import json

from app.providers.llm import ChatMessage, get_llm_provider

SYNTHESIS_SYSTEM = """You are a rigorous energy analyst synthesizing a final answer.

You are given:
1. VERIFIED_SQL — structured price data and deterministic statistics. These are the ONLY allowed numbers.
2. EVIDENCE — ranked retrieved passages, each with an index like [0], [1], ...

Rules:
- State numbers exactly as they appear in VERIFIED_SQL. Do NOT round, invent, or combine them in ways not shown.
- If a number is absent from VERIFIED_SQL, do not provide it. Say the data is unavailable.
- Support every historical/causal claim with a citation to EVIDENCE using its index in square brackets, e.g. [1].
- Do not cite a number to EVIDENCE; numbers are backed by SQL only.
- If evidence conflicts, say so and prefer the SQL number for quantities.
- Be concise and factual.
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
                    "source": c.get("source"),
                    "title": c.get("title"),
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
            "Write the final cited answer."
        )

        messages = [
            ChatMessage(role="system", content=SYNTHESIS_SYSTEM),
            ChatMessage(role="user", content=user),
        ]
        response = self.llm.chat(messages, temperature=0.0)
        return response.get("content", "")
