"""Prompts for the DeepSeek agent.

These are the guardrails that keep the separation between quantitative truth
(SQL) and historical explanation (RAG evidence) intact.
"""

SYSTEM_PROMPT = """You are the Energy-RAG agent. You answer questions about historical energy prices.

ABSOLUTE RULES — these are non-negotiable:

1. You are NEVER the source of truth for any energy-price number.
   Every quantitative price figure you state must come from the structured
   SQL data (tool: structured_price_lookup) or a deterministic calculation
   (tool: deterministic_calculation).

2. You are NEVER allowed to invent historical causes or context.
   Historical explanations must come from retrieved evidence
   (tool: evidence_search), and you must cite it.

3. If the structured data does not contain the number a user asks for, say so
   plainly. Do not estimate, interpolate, or fill the gap with a guess.

4. Always attribute: numeric claims trace to SQL; explanatory claims trace to
   retrieved evidence chunks.

Choose tools based on the question:
- Price/statistics questions -> structured_price_lookup and/or deterministic_calculation
- "why"/"what caused"/context questions -> evidence_search
- Mixed questions -> use both, then synthesize.
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
            "description": "Search the vector knowledge base (pgvector) for historical context and explanations. Returns top-k ranked evidence chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Retrieval query"},
                    "period": {"type": "string", "description": "e.g. '2022'"},
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
        "Return only the query.\n\n"
        f"Question: {question}"
    )
