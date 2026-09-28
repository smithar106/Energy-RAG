"""Provider abstractions.

The two external-model boundaries are wrapped behind interfaces so that either
model (the hosted LLM or the local embedding model) can be swapped without
rewriting the agent.
"""
