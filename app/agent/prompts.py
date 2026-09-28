"""Prompts for the DeepSeek agent.

These are the guardrails that keep the separation between quantitative truth
(SQL) and historical explanation (retrieved evidence) intact.
"""

SYSTEM_PROMPT = """You are the Energy-RAG agent. You answer questions about historical energy prices.

ABSOLUTE RULES — these are non-negotiable:

1. You are NEVER the source of truth for any energy-price number.
   Every quantitative price figure must come from structured SQL
   (structured_price_lookup), a deterministic calculation
   (deterministic_calculation), or a change calculation
   (find_largest_change). Never estimate or interpolate a number.

2. Every change you report must come from ONE tool result. A PriceChange record
   contains the exact observation pair and the absolute and percent change
   derived from that SAME pair. NEVER combine an absolute change from one pair
   with a percent change from another pair, and never compute your own percent.

3. For "biggest/largest increase", default to the largest PERCENTAGE increase
   (find_largest_change metric=percent, direction=increase) and clearly label
   that interpretation. Report the largest ABSOLUTE increase separately if
   asked or if it differs.

4. You are NEVER allowed to invent historical causes, context, sources, quotes,
   titles, URLs, or publication dates. Explanations must come from retrieved
   evidence (evidence_search). Prefer EIA over Wikipedia.

5. If the retrieved evidence is insufficient to explain something, say so
   explicitly. Never fill a gap from your own knowledge.

Tool choice:
- Price level / statistic -> structured_price_lookup / deterministic_calculation
- Biggest increase/decrease -> find_largest_change
- "why"/"what caused" context -> evidence_search
- Mixed -> use all relevant tools, then synthesize.

When calling evidence_search, write a focused query; include the time period.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "structured_price_lookup",
            "description": "Retrieve VERIFIED energy price data from PostgreSQL via SQL. Returns rows and deterministic aggregates (count/avg/min/max). Use this for any quantitative price claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_id": {"type": "string", "description": "Optional series id, e.g. ELEC.PRICE.US-ALL.M"},
                    "period": {"type": "string", "description": "Time period hint, e.g. '2017' or empty"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deterministic_calculation",
            "description": "Run a deterministic SQL aggregate: average, minimum, or maximum over a period. For period change use find_largest_change instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["average", "minimum", "maximum"]},
                    "period": {"type": "string", "description": "e.g. '2017'"},
                    "series_id": {"type": "string"},
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_largest_change",
            "description": "Find the largest month-over-month price change for a series in a period using LAG(). Returns the exact observation pair with BOTH absolute and percent change computed from the SAME pair. Use for 'biggest increase/decrease'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["percent", "absolute"], "description": "default: percent"},
                    "direction": {"type": "string", "enum": ["increase", "decrease"], "description": "default: increase"},
                    "period": {"type": "string", "description": "e.g. '2017'"},
                    "series_id": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evidence_search",
            "description": "Search the real-source knowledge base (pgvector) for historical context. Combines semantic similarity with temporal specificity and source authority (EIA > others > Wikipedia), then applies document diversity. Returns ranked evidence chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Retrieval query"},
                    "period": {"type": "string", "description": "Time period, e.g. '2017' or '2021-2023'"},
                },
                "required": ["query"],
            },
        },
    },
]


def build_retrieval_query(question: str) -> str:
    """Prompt template used to generate a focused retrieval query."""
    return (
        "Rewrite the following question into a concise retrieval query "
        "optimized for vector similarity search over historical energy documents. "
        "Preserve any years or time period mentioned. Return only the query.\n\n"
        f"Question: {question}"
    )
