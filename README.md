# Energy-RAG

A **hybrid retrieval RAG agent** for answering questions about historical
energy prices. It keeps a strict separation between **quantitative truth**
(SQL over structured EIA price data) and **historical explanation** (retrieved
document evidence), then synthesizes a grounded, cited answer.

- **LLM:** [DeepSeek](https://www.deepseek.com) (the *only* hosted LLM; used via
  its OpenAI-compatible API at `https://api.deepseek.com`).
- **Embeddings:** `BAAI/bge-small-en-v1.5` via `sentence-transformers`, generated
  **locally** — no paid embeddings API.
- **Vector store:** PostgreSQL + `pgvector`.
- **Structured data:** U.S. EIA Open Data API (authoritative prices).

## Architecture

```
                USER QUESTION
                      |
                      v
               DEEPSEEK AGENT            (tool selection / orchestration)
                      |
          +-----------+-----------+
          |                       |
          v                       v
   STRUCTURED DATA              RAG
          |                       |
        SQL                 BGE EMBEDDINGS
          |                       |
     PostgreSQL               PGVECTOR
          |                       |
   VERIFIED NUMBERS       RETRIEVED EVIDENCE
          |                       |
          +-----------+-----------+
                      |
                      v
            DEEPSEEK SYNTHESIS        (grounded answer + citations)
                      |
                      v
            GROUNDING VALIDATOR       (output validation)
                      |
                      v
                CITED RESPONSE
```

### Hard rules

1. **DeepSeek is never the source of truth for a number.** Every quantitative
   price claim originates from SQL against `price_records`.
2. **DeepSeek never invents historical causes.** Explanations originate from
   retrieved RAG evidence and are cited.
3. If the structured data lacks a requested number, the agent says so rather
   than estimating.

## The ten visible components

| # | Component | Where | What it proves |
|---|-----------|-------|----------------|
| 1 | Structured SQL retrieval | `app/retrieval/sql.py` | Verified numbers come from SQL, not the LLM |
| 2 | Deterministic calculations | `app/retrieval/sql.py::deterministic_calculation` | avg/min/max/change/% computed in SQL |
| 3 | Vector embeddings | `app/providers/embeddings.py` | local `bge-small-en-v1.5` → 384-dim vectors |
| 4 | pgvector similarity search | `app/retrieval/vector.py` | cosine search `<=>` over `chunks.embedding` |
| 5 | Temporal filtering | `app/retrieval/temporal.py` | period parsing + date-bound filters on SQL & vectors |
| 6 | Evidence ranking | `app/retrieval/reranker.py` | similarity + lexical + temporal reranking |
| 7 | DeepSeek agent/tool orchestration | `app/agent/orchestrator.py` | tool-calling loop, tools chosen by DeepSeek |
| 8 | Grounded synthesis | `app/synthesis/synthesizer.py` | DeepSeek restricted to SQL numbers + evidence |
| 9 | Source attribution | `app/synthesis/citations.py` | `[n]` markers map to real chunks/documents |
| 10 | Output validation | `app/synthesis/grounding.py` | numeric + explanatory grounding checks |

## Data model

- **`price_records`** — structured truth. `series_id`, `period` (temporal axis),
  `price`, `units`, `region`, `fuel`, `source`.
- **`documents` / `chunks`** — unstructured evidence. Chunks hold a
  `vector(384)` embedding plus `start_date`/`end_date` temporal bounds.

## Environment variables

See `.env.example`. Required:

```
DEEPSEEK_API_KEY=     # DeepSeek API key (the only hosted LLM)
EIA_API_KEY=          # U.S. EIA Open Data key
DATABASE_URL=         # postgresql+psycopg://user:pass@host:5432/energy_rag
```

Optional / configuration:

```
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

Never commit real credentials — `.env` is git-ignored.

## Local development

```bash
# 1. Start PostgreSQL with pgvector
docker compose up -d db

# 2. Create a virtualenv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 3. Seed demo data (structured prices + a sample document)
python -m scripts.seed_demo

# 4. Run the API (loads the embedding model once at startup)
uvicorn app.main:app --reload
```

Query it:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was the average U.S. electricity price in 2022, and why did it change?"}'
```

### Ingestion

```bash
# Ingest a local document into the knowledge base
python -m scripts.ingest --title "..." --source wikipedia \
    path/to/file.txt --start 2022-01-01 --end 2023-01-01

# Pull a real EIA series into the structured table
python -m scripts.fetch_eia --series ELEC.PRICE.US-ALL.M --length 5000
```

The ingestion pipeline is: **document → clean → chunk → local embedding →
pgvector**. Retrieval is: **question + time period → retrieval query → local
embedding → pgvector similarity search → top-k → rerank → DeepSeek synthesis**.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | liveness + model/provider info |
| `POST` | `/ask` | run the full hybrid pipeline |
| `POST` | `/ingest` | ingest a document (clean → chunk → embed → pgvector) |
| `POST` | `/ingest/eia` | fetch + store an EIA series |

## Deployment (Railway)

The `Dockerfile` bakes the embedding model into the image so it is **not
re-downloaded on cold start**, and `railway.json` wires up the build + health
check. Railway auto-provisions a PostgreSQL database; the `pgvector` extension
is created at startup by `app.db.base.init_db()`.

1. Create a Railway project from this repo.
2. Add a PostgreSQL service (enable the `vector` extension if required) and
   reference the generated `DATABASE_URL`.
3. Set `DEEPSEEK_API_KEY`, `EIA_API_KEY` as service variables.

> If deployment size/memory becomes an issue, embeddings can later be isolated
> into a separate service. That change is intentionally deferred — the
> `EmbeddingProvider` interface makes it a drop-in swap.

## Tests

```bash
python -m pytest
```

Unit tests cover chunking, cleaning, temporal parsing/filtering, reranking, and
grounding — no database or network required.

## Provider abstraction

Both model boundaries are behind interfaces so they can be swapped without
touching the agent:

- `app/providers/llm.py` — `LLMProvider` / `DeepSeekProvider` (OpenAI-compatible
  client pointed at `https://api.deepseek.com`).
- `app/providers/embeddings.py` — `EmbeddingProvider` /
  `SentenceTransformerEmbeddings` (process-wide singleton, loaded once).
