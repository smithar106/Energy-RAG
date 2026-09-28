"""Prompts for the DeepSeek agent.

These are the guardrails that keep the separation between quantitative truth
(SQL) and historical explanation (retrieved evidence) intact.
"""

SYSTEM_PROMPT = """You are the Energy-RAG agent. You answer questions about historical energy prices.

ABSOLUTE RULES — these are non-negotiable:

1. You are NEVER the source of truth for any energy-price number.
   Every quantitative price figure must come from structured SQL
   (tool: structured_price_lookup) or a deterministic calculation
   (tool: deterministic_calculation). Never estimate or interpolate a number.

2. You are NEVER allowed to invent historical causes, context, sources, quotes,
   titles, URLs, or publication dates. Historical explanations must come from
   retrieved evidence (tool: evidence_search).

3. Prefer authoritative sources: EIA analysis outranks Wikipedia. If both are
   retrieved, lean on EIA.

4. If the retrieved evidence is insufficient to explain something, say so
   explicitly. Never fill a gap from your own knowledge.

Choose tools based on the question:
- Price / statistic questions -> structured_price_lookup and/or deterministic_calculation
- "why" / "what caused" / context questions -> evidence_search
- Mixed questions -> use both, then synthesize.

When calling evidence_search, write a focused retrieval query and include the
time period from the question (e.g. "2021-2023", "2022", "2015").
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
                    "series_id": {
                        "type": "string",
                        "description": "Optional series id, e.g. ELEC.PRICE.US-ALL.M",
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period hint, e.g. '2022' or '2022-03' or empty",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deterministic_calculation",
            "description": "Run a deterministic SQL computation on verified price data: average, minimum, maximum, change, or pct_change over a period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["average", "minimum", "maximum", "change", "pct_change"],
                    },
                    "period": {"type": "string", "description": "e.g. '2022'"},
                    "series_id": {"type": "string"},
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evidence_search",
            "description": "Search the real-source knowledge base (pgvector) for historical context and explanations. Combines semantic similarity with temporal relevance and source authority (EIA > Wikipedia). Returns ranked evidence chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Retrieval query"},
                    "period": {"type": "string", "description": "Time period, e.g. '2021-2023' or '2022'"},
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
