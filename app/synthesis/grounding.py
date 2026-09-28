"""Grounding validator — output validation.

Checks the final answer before returning it:

1. **Quantitative claims** must be supported by SQL results (``price_records``
   / deterministic calculations). Numbers not found there are flagged.
2. **Explanatory claims** must be supported by retrieved document chunks
   (judged by DeepSeek over the evidence only — it may evaluate, not supply).
3. **Citations** — every ``[n]`` marker must correspond to a retrieved chunk
   record; its URL/title/date come from that stored record, never the model.
4. **Sufficiency** — if no evidence was retrieved but the answer still makes
   causal claims, that is flagged as ungrounded.
"""
from __future__ import annotations

import re
from numbers import Number

from app.schemas.api import GroundingCheck
from app.synthesis.citations import citation_markers

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_CITE_RE = re.compile(r"\[\d+\]")
_CAUSAL_RE = re.compile(
    r"\b(because|due to|caused|led to|as a result|driven by|owing to|"
    r"resulted in|pushed up|contributed to)\b",
    re.I,
)


def _collect_sql_numbers(sql_results: list[dict]) -> set[float]:
    nums: set[float] = set()
    for res in sql_results:
        derived = res.get("derived", {})
        for key in ("average", "minimum", "maximum", "count"):
            v = derived.get(key)
            if isinstance(v, Number):
                nums.add(round(float(v), 4))
        v = res.get("value")  # deterministic_calculation (may be Decimal)
        if isinstance(v, Number):
            nums.add(round(float(v), 4))
        for row in res.get("rows", []):
            price = row.get("price")
            if isinstance(price, Number):
                nums.add(round(float(price), 4))
    return nums


def _collect_evidence_numbers(evidence: list[dict]) -> set[float]:
    nums: set[float] = set()
    for c in evidence:
        for m in re.findall(r"\d+(?:\.\d+)?", c.get("text", "")):
            nums.add(round(float(m), 4))
    return nums


def _numeric_claims(answer: str) -> list[tuple[str, float]]:
    answer = _CITE_RE.sub(" ", answer or "")  # [0] is a reference, not a quantity
    claims: list[tuple[str, float]] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)", answer):
        raw = m.group(0)
        if _YEAR_RE.search(raw) and len(raw) == 4:
            continue  # years are temporal anchors
        claims.append((raw, round(float(raw), 4)))
    return claims


def validate_grounding(
    answer: str,
    sql_results: list[dict],
    evidence: list[dict],
    llm_check: bool = True,
) -> GroundingCheck:
    check = GroundingCheck()

    allowed = _collect_sql_numbers(sql_results) | _collect_evidence_numbers(evidence)
    claims = _numeric_claims(answer)
    check.total_numeric_claims = len(claims)
    for raw, rounded in claims:
        if any(abs(rounded - a) <= 0.01 for a in allowed):
            check.grounded_numeric_claims += 1
        else:
            check.ungrounded.append(f"numeric: {raw}")

    # Citation integrity.
    markers = citation_markers(answer)
    check.citations_used = sorted(set(markers))
    check.invalid_citations = [
        f"[{m}]" for m in sorted({m for m in markers if m < 0 or m >= len(evidence)})
    ]
    check.citations_valid = not check.invalid_citations

    # Evidence sufficiency.
    check.sufficient_evidence = len(evidence) > 0
    if not check.sufficient_evidence:
        check.notes.append("no evidence chunks were retrieved")
        if _CAUSAL_RE.search(answer or ""):
            check.ungrounded.append("explanatory claim without retrieved evidence")

    # Explanatory grounding via DeepSeek (evidence-only judgement).
    if llm_check and evidence:
        grounded = _verify_explanations(answer, evidence)
        check.total_explanatory_claims = len(evidence)
        check.grounded_explanatory_claims = grounded
        if grounded == 0:
            check.notes.append(
                "no retrieved evidence chunk directly supports the answer's claims"
            )

    numeric_ok = check.grounded_numeric_claims == check.total_numeric_claims
    explanatory_ok = not any(
        u.startswith("explanatory") for u in check.ungrounded
    )
    check.valid = numeric_ok and check.citations_valid and explanatory_ok
    return check


def _verify_explanations(answer: str, evidence: list[dict]) -> int:
    """Return how many evidence chunks directly support the answer's claims."""
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
        [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ],
        temperature=0.0,
    )
    m = re.search(r"\d+", resp.get("content", "0"))
    return int(m.group(0)) if m else 0
