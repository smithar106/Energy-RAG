# Energy-RAG

A **hybrid-retrieval RAG agent** for answering questions about historical
energy prices. It keeps a strict separation between **quantitative truth**
(SQL over structured EIA price data) and **historical explanation** (retrieved
*real* source documents), then synthesizes a grounded, cited answer.

- **LLM:** [DeepSeek](https://www.deepseek.com) (the *only* hosted LLM; used via
  its OpenAI-compatible API at `https://api.deepseek.com`).
- **Embeddings:** `BAAI/bge-small-en-v1.5` via `sentence-transformers`, generated
  **locally** — no paid embeddings API.
- **Vector store:** PostgreSQL + `pgvector`.
- **Structured data:** U.S. EIA Open Data API (authoritative prices).
- **RAG sources:** EIA *Today in Energy* analysis (primary) + Wikipedia
  (secondary). **No synthetic content.**

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
            DEEPSEEK SYNTHESIS        (grounded answer + [n] citations)
                      |
                      v
            GROUNDING VALIDATOR       (numbers / explanations / citations)
                      |
                      v
                CITED RESPONSE
```

### Hard rules

1. **DeepSeek is never the source of truth for a number.** Every quantitative
   price claim originates from SQL against `price_records`.
2. **DeepSeek never invents historical causes, sources, quotes, titles, URLs,
   or dates.** Explanations come from retrieved chunks; citation metadata is
   injected programmatically from the stored records.
3. If the structured data lacks a requested number, the agent says so.
4. If retrieval finds insufficient evidence, the agent says so — it does not
   answer from model memory.

## The ten visible components

| # | Component | Where |
|---|-----------|-------|
| 1 | Structured SQL retrieval | `app/retrieval/sql.py` |
| 2 | Deterministic calculations | `app/retrieval/sql.py::deterministic_calculation` |
| 3 | Vector embeddings (local BGE) | `app/providers/embeddings.py` |
| 4 | pgvector similarity search | `app/retrieval/vector.py` |
| 5 | Temporal filtering | `app/retrieval/temporal.py`, `ranking.py` |
| 6 | Evidence ranking | `app/retrieval/ranking.py` |
| 7 | DeepSeek agent/tool orchestration | `app/agent/orchestrator.py` |
| 8 | Grounded synthesis | `app/synthesis/synthesizer.py` |
| 9 | Source attribution | `app/synthesis/citations.py` |
| 10 | Output validation | `app/synthesis/grounding.py` |

Plus a **RAG trace** (`/ask` → `trace`, `/debug/retrieval`) exposing the whole
retrieval path.

## Data model

- **`price_records`** — structured truth (real EIA data). `series_id`, `period`,
  `price`, `units`, `region`, `fuel`, `source`.
- **`documents`** — one real source page. `title`, `source_name`, `source_type`,
  `source_url` (unique), `published_date`, `revision_id`, `retrieved_at`.
- **`chunks`** — verbatim source text + `vector(384)` embedding, plus
  denormalized provenance (`source_name`, `source_type`, `document_title`,
  `source_url`, `published_date`) and the **event window** (`start_year`,
  `end_year`).

`published_date` (when the source was published) and `start_year`/`end_year`
(the event window the source discusses) are stored **separately** — publication
year is not assumed to equal event year.

## Retrieval ranking (inspectable)

```
final_score = 0.60 * semantic_similarity      # pgvector cosine, [0,1]
            + 0.25 * temporal_score           # event-window overlap with the query period
            + 0.15 * source_authority         # EIA = 1.0, Wikipedia = 0.6, unknown = 0.5
```

- `temporal_score` is neutral (0.5) when the query has no period or the chunk
  has no event window; rises to 1.0 as the chunk's event window covers the
  query period; decays toward 0 as the gap grows.
- `source_authority` breaks ties toward EIA over Wikipedia.

Weights and every component score are returned in the RAG trace — nothing is
hidden in a framework.

## Environment variables

```
DEEPSEEK_API_KEY=       # DeepSeek (only hosted LLM)
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/energy_rag
EIA_API_KEY=            # U.S. EIA Open Data
```

Never commit real credentials — `.env` is git-ignored.

## Local development

```bash
docker compose up -d db
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/` | landing page |
| `GET`  | `/health` | liveness + provider info |
| `POST` | `/ask` | full pipeline → cited answer + RAG trace |
| `POST` | `/debug/retrieval` | retrieval only, with ranking scores |
| `POST` | `/ingest/eia` | ingest an EIA price series (structured) |
| `POST` | `/ingest` | ingest a caller-supplied real document |
| `POST` | `/admin/ingest/eia` | ingest EIA Today in Energy articles (RAG) |
| `POST` | `/admin/ingest/wikipedia` | ingest a Wikipedia article (RAG) |
| `POST` | `/admin/reset-rag` | drop+recreate RAG tables (keeps `price_records`) |
| `GET`  | `/admin/stats` | knowledge-base verification |

## Ingestion

Sources are real and fetched live — **no generated or summarized text is ever
stored**; each chunk is verbatim source content.

```bash
# EIA Today in Energy / analysis (primary source)
python -m scripts.ingest_eia_documents --limit 20

# Wikipedia (secondary), curated seed list or specific titles
python -m scripts.ingest_wikipedia
python -m scripts.ingest_wikipedia --title "Shale gas"

# Remove synthetic/placeholder RAG content (never touches price_records)
python -m scripts.reset_rag_content

# Verify the KB + run retrieval tests
python -m scripts.verify_db
```

Every script also supports `--api-url https://<host>` to run against the
deployed service.

Pipeline: **real source → clean → section-aware chunk → local embedding →
pgvector**. Chunking targets ~500–800 tokens (~550 words) with ~100-token
(~80-word) overlap and respects section / paragraph / sentence boundaries.

## Deployment (Railway)

The `Dockerfile` bakes the embedding model into the image so it is not
re-downloaded on cold start; `railway.json` wires the build + health check.
Railway provisions PostgreSQL; `pgvector` is created at startup.

1. Create a Railway project from this repo + a PostgreSQL service.
2. Set `DEEPSEEK_API_KEY`, `EIA_API_KEY`; reference `DATABASE_URL`.
3. Deploy, then run ingestion (via the admin endpoints / scripts).

## Tests

```bash
python -m pytest
```

Unit tests cover chunking, HTML/section extraction, year inference, temporal
filtering, hybrid ranking, and grounding — no DB or network required.
