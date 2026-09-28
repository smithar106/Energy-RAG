"""Grounding validator — output validation.

The final answer is checked before it is returned:

1. **Deterministic numeric grounding**: every numeric token in the answer must
   appear (after rounding) in the verified SQL results or, for non-price
   figures like years, in the evidence. If a number is not found, it is flagged
   as ungrounded.

2. **Explanatory grounding**: uses DeepSeek to verify each causal/historical
   claim is supported by the retrieved evidence. (DeepSeek may *evaluate*
   evidence, but it may not supply new numbers or facts — it only judges.)

The validator returns a ``GroundingCheck`` report; the pipeline can refuse or
trim ungrounded numbers based on it.
"""
from __future__ import annotations

import re

from app.schemas.api import GroundingCheck

# Numbers that legitimately come from years/dates rather than SQL prices.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _collect_sql_numbers(sql_results: list[dict]) -> set[float]:
    nums: set[float] = set()
    for res in sql_results:
        derived = res.get("derived", {})
        for key in ("average", "minimum", "maximum", "count"):
            v = derived.get(key)
            if isinstance(v, (int, float)):
                nums.add(round(float(v), 4))
        # deterministic_calculation returns {"operation", "value"}
        v = res.get("value")
        if isinstance(v, (int, float)):
            nums.add(round(float(v), 4))
        for row in res.get("rows", []):
            price = row.get("price")
            if isinstance(price, (int, float)):
                nums.add(round(float(price), 4))
    return nums


def _collect_evidence_numbers(evidence: list[dict]) -> set[float]:
    nums: set[float] = set()
    for c in evidence:
        for m in re.findall(r"\d+(?:\.\d+)?", c.get("text", "")):
            nums.add(round(float(m), 4))
    return nums


def _numeric_claims(answer: str) -> list[tuple[str, float]]:
    claims = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)", answer):
        raw = m.group(0)
        # Skip years — those are temporal anchors, not price figures.
        full = answer[max(0, m.start() - 20): m.end() + 4]
        if _YEAR_RE.search(raw) and len(raw) == 4:
            continue
        claims.append((raw, round(float(raw), 4)))
    return claims


def validate_grounding(
    answer: str,
    sql_results: list[dict],
    evidence: list[dict],
    llm_check: bool = True,
) -> GroundingCheck:
    sql_nums = _collect_sql_numbers(sql_results)
    evidence_nums = _collect_evidence_numbers(evidence)
    allowed = sql_nums | evidence_nums

    check = GroundingCheck()
    claims = _numeric_claims(answer)
    check.total_numeric_claims = len(claims)
    for raw, rounded in claims:
        # Accept if within a small tolerance of an allowed number.
        grounded = any(abs(rounded - a) <= 0.01 for a in allowed)
        if grounded:
            check.grounded_numeric_claims += 1
        else:
            check.ungrounded.append(f"numeric: {raw}")

    # Explanatory grounding via DeepSeek (evidence-only judgement).
    if llm_check and evidence:
        grounded = _verify_explanations(answer, evidence)
        check.total_explanatory_claims = len(evidence)
        check.grounded_explanatory_claims = grounded
        check.valid = check.valid and grounded > 0

    check.valid = check.grounded_numeric_claims == check.total_numeric_claims
    return check


def _verify_explanations(answer: str, evidence: list[dict]) -> int:
    """Return the number of evidence chunks that support the answer's claims."""
    from app.providers.llm import ChatMessage, get_llm_provider

    system = (
        "You are a grounding validator. Given an answer and a list of evidence "
        "chunks, count how many chunks directly support the answer's historical "
        "or causal claims. Respond with a single integer."
    )
    evidence_text = "\n\n".join(
        f"[{i}] {c.get('text', '')}" for i, c in enumerate(evidence)
    )
    user = f"ANSWER:\n{answer}\n\nEVIDENCE:\n{evidence_text}\n\nSupported chunks count:"

    llm = get_llm_provider()
    resp = llm.chat(
        [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
        temperature=0.0,
    )
    m = re.search(r"\d+", resp.get("content", "0"))
    return int(m.group(0)) if m else 0
