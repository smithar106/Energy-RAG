"""DeepSeek tool-calling loop.

The agent *gathers* evidence (SQL + pgvector) by calling tools; it never
computes a number or invents a source itself. Grounded synthesis happens
downstream in the ``synthesis`` module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS
from app.agent.tools import execute_tool
from app.providers.llm import ChatMessage, get_llm_provider

MAX_ITERATIONS = 6


@dataclass
class AgentResult:
    tool_calls: list[dict] = field(default_factory=list)
    sql_results: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)          # full ranked records
    retrieval_trace: dict | None = None
    retrieval_query: str | None = None


class AgentOrchestrator:
    def __init__(self) -> None:
        self.llm = get_llm_provider()

    def run(self, question: str) -> AgentResult:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=question),
        ]

        sql_results: list[dict] = []
        evidence: list[dict] = []
        calls_log: list[dict] = []
        retrieval_trace: dict | None = None
        retrieval_query: str | None = None

        for _ in range(MAX_ITERATIONS):
            response = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                break

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.get("content", ""),
                    tool_calls=tool_calls,
                )
            )

            for call in tool_calls:
                name = call["name"]
                arguments = call["arguments"]
                result = execute_tool(name, arguments, sql_results)
                calls_log.append({"name": name, "arguments": arguments})

                if name in {"structured_price_lookup", "deterministic_calculation"}:
                    sql_results.append(result)
                elif name == "evidence_search":
                    trace = result.pop("_trace", None)
                    if trace:
                        retrieval_trace = trace
                        retrieval_query = trace.get("retrieval_query")
                        # Only accepted evidence is passed to synthesis.
                        evidence = [c for c in trace.get("chunks", []) if c.get("accepted")]

                # The result sent to the model excludes the private trace.
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(result, default=str),
                        tool_call_id=call["id"],
                        name=name,
                    )
                )

        return AgentResult(
            tool_calls=calls_log,
            sql_results=sql_results,
            evidence=evidence,
            retrieval_trace=retrieval_trace,
            retrieval_query=retrieval_query,
        )
