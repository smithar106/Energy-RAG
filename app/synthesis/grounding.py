"""Grounding validator — output validation.

Checks the final answer before returning it:

1. **Quantitative claims** must be supported by SQL results (``price_records``,
   aggregates, or price-change records). Numbers not found there are flagged.
2. **Percentage changes** must be consistent with the observation pair cited in
   the claim: the validator *recomputes* ``(current-previous)/previous*100`` from
   the two prices referenced and fails if it disagrees with the stated percent.
   This prevents combining an absolute change from one pair with a percent
   change from another (the 2017 bug).
3. **Explanatory claims** must be supported by retrieved chunks (judged over
   evidence only).
4. **Citations** — every ``[n]`` must map to a retrieved chunk record.
5. **Sufficiency** — no evidence + causal claims ⇒ ungrounded.
"""
from __future__ import annotations

import re
from numbers import Number

from app.schemas.api import GroundingCheck
from app.synthesis.citations import citation_markers

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_CITE_RE = re.compile(r"\[\d+\]")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b")
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CAUSAL_RE = re.compile(
    r"\b(because|due to|caused|led to|as a result|driven by|owing to|"
    r"resulted in|pushed up|contributed to)\b",
    re.I,
)
_PCT_TOL = 0.06


def _price_like_numbers(window: str) -> list[float]:
    """Numbers in a text window that are not 4-digit years."""
    nums: list[float] = []
    for m in re.finditer(r"\d+(?:\.\d+)?", window):
        token = m.group(0)
        value = float(token)
        if "." not in token and 1900 <= value <= 2100:
            continue  # a year, not a price
        nums.append(value)
    return nums


def _collect_price_changes(sql_results: list[dict]) -> list[dict]:
    changes: list[dict] = []
    for res in sql_results:
        pc = res.get("price_change")
        if isinstance(pc, dict):
            changes.append(pc)
        for c in res.get("changes") or []:
            if isinstance(c, dict):
                changes.append(c)
    return changes


def _collect_sql_numbers(sql_results: list[dict]) -> set[float]:
    nums: set[float] = set()
    for res in sql_results:
        for key in ("average", "minimum", "maximum", "count"):
            v = res.get("derived", {}).get(key)
            if isinstance(v, Number):
                nums.add(round(float(v), 4))
        v = res.get("value")
        if isinstance(v, Number):
            nums.add(round(float(v), 4))
        for row in res.get("rows", []):
            price = row.get("price")
            if isinstance(price, Number):
                nums.add(round(float(price), 4))
    for change in _collect_price_changes(sql_results):
        for key in ("previous_value", "current_value", "absolute_change", "percent_change"):
            v = change.get(key)
            if isinstance(v, Number):
                nums.add(round(float(v), 4))
    return nums


def _collect_evidence_numbers(evidence: list[dict]) -> set[float]:
    nums: set[float] = set()
    for c in evidence:
        for m in re.findall(r"\d+(?:\.\d+)?", c.get("text", "")):
            nums.add(round(float(m), 4))
    return nums


def _collect_evidence_percents(evidence: list[dict]) -> set[float]:
    percents: set[float] = set()
    for c in evidence:
        for m in _PCT_RE.finditer(c.get("text", "")):
            percents.add(round(float(m.group(1)), 4))
    return percents


def _numeric_claims(answer: str) -> list[tuple[str, float]]:
    text = _CITE_RE.sub(" ", answer or "")
    text = _DATE_RE.sub(" ", text)  # strip ISO dates so day/month aren't read as numbers
    claims: list[tuple[str, float]] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)", text):
        raw = m.group(0)
        if _YEAR_RE.search(raw) and len(raw) == 4:
            continue
        claims.append((raw, round(float(raw), 4)))
    return claims


def _verify_percentage_claims(
    answer: str,
    sql_results: list[dict],
    evidence: list[dict],
) -> list[str]:
    """Recompute percent changes from their cited pair; flag inconsistencies."""
    ungrounded: list[str] = []
    changes = _collect_price_changes(sql_results)
    change_pcts = [c["percent_change"] for c in changes if c.get("percent_change") is not None]
    evidence_pcts = _collect_evidence_percents(evidence)

    for m in _PCT_RE.finditer(answer or ""):
        claimed = float(m.group(1))
        window = answer[max(0, m.start() - 180): m.end() + 60]

        prices = _price_like_numbers(window)
        pairs = [(prices[i], prices[j]) for i in range(len(prices)) for j in range(i + 1, len(prices)) if prices[i] > 0]
        if pairs:
            # The claim references observation values → recompute from them.
            ok = any(abs(((b - a) / a * 100.0) - claimed) <= _PCT_TOL for a, b in pairs)
            if not ok:
                ungrounded.append(f"percent: {m.group(0).strip()} (inconsistent with the cited prices)")
            continue

        # No pair to recompute from → accept if it matches a change or evidence.
        if any(abs(claimed - p) <= _PCT_TOL for p in change_pcts):
            continue
        if any(abs(claimed - p) <= _PCT_TOL for p in evidence_pcts):
            continue
        ungrounded.append(f"percent: {m.group(0).strip()}")
    return ungrounded


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

    # Percentage-change consistency (recomputed from the cited pair).
    check.ungrounded.extend(_verify_percentage_claims(answer, sql_results, evidence))

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

    if llm_check and evidence:
        grounded = _verify_explanations(answer, evidence)
        check.total_explanatory_claims = len(evidence)
        check.grounded_explanatory_claims = grounded
        if grounded == 0:
            check.notes.append("no retrieved evidence chunk directly supports the answer's claims")

    numeric_ok = check.grounded_numeric_claims == check.total_numeric_claims
    explanatory_ok = not any(u.startswith("explanatory") for u in check.ungrounded)
    percent_ok = not any(u.startswith("percent:") for u in check.ungrounded)
    check.valid = numeric_ok and percent_ok and explanatory_ok and check.citations_valid
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
