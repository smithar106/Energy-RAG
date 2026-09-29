"""Retrieval evaluation harness.

Runs a set of historical retrieval questions against the deployed service and
reports per-question metrics (candidates → gate-passed → accepted evidence) so
the irrelevant-evidence acceptance rate can be inspected directly.

    python -m scripts.eval_retrieval --api-url https://<host>
"""
from __future__ import annotations

import argparse
from collections import Counter

import httpx

QUESTIONS = [
    "Why did U.S. electricity prices rise between 2021 and 2023?",
    "What caused the biggest increase in US electricity prices in 2017 and what was the % increase?",
    "Why did energy prices decline around 2015?",
    "What factors affected energy prices during COVID-19?",
    "What happened to U.S. natural gas prices in 2022?",
    "Why did energy prices spike in 2008?",
    "What happened to U.S. electricity prices in 2010?",
    "Why did crude oil prices fall in 2014 and 2015?",
    "What drove gasoline prices in 2007?",
    "How did weather affect U.S. electricity prices in 2023?",
    "Why did coal generation decline in the 2010s?",
    "What caused natural gas prices to rise in 2005?",
    "How did U.S. electricity prices change between 2008 and 2009?",
    "What affected retail electricity prices in 2019?",
    "Why did U.S. natural gas production grow in 2024?",
    "What caused the 1970s energy crisis?",
    "How did LNG exports affect U.S. gas prices in 2025?",
    "What drove electricity demand in 2024?",
    "Why did U.S. electricity prices rise in 2016?",
    "What happened to energy prices in 2026?",
]


def run(api_url: str, question: str) -> dict:
    r = httpx.post(
        api_url.rstrip("/") + "/debug/retrieval",
        json={"query": question},
        timeout=240,
    )
    r.raise_for_status()
    trace = r.json()["trace"]
    chunks = trace.get("chunks", [])
    return {
        "candidates": trace.get("candidate_count", 0),
        "passed": sum(1 for c in chunks if c.get("gate") == "PASS"),
        "rejected": sum(1 for c in chunks if c.get("gate") != "PASS"),
        "reasons": Counter(c.get("failure_reason") for c in chunks if c.get("gate") != "PASS"),
        "top_sources": [c.get("source_name") for c in chunks if c.get("gate") == "PASS"][:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    args = parser.parse_args()

    total_candidates = 0
    total_passed = 0
    rows = []
    for q in QUESTIONS:
        r = run(args.api_url, q)
        total_candidates += r["candidates"]
        total_passed += r["passed"]
        rows.append((q, r))

    print(f"{'question':<70} {'cand':>5} {'pass':>5} {'rej':>5}  top sources")
    for q, r in rows:
        src = ", ".join(dict.fromkeys(r["top_sources"]))
        print(f"{q[:68]:<70} {r['candidates']:>5} {r['passed']:>5} {r['rejected']:>5}  {src[:48]}")

    print(f"\ntotal candidates: {total_candidates}")
    print(f"gate-passed (avg per question): {total_passed / len(rows):.1f}")
    print(f"gate-pass rate: {total_passed / max(1, total_candidates) * 100:.1f}%")
    print("(low gate-pass rate => tight evidence gate => low irrelevant-acceptance rate)")


if __name__ == "__main__":
    main()
