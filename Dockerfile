# Energy-RAG — hybrid retrieval RAG agent
#
# Build strategy (Railway-friendly):
#   1. Install deps (torch + sentence-transformers are the heavy part).
#   2. Pre-download the embedding model into the image so it is baked in and
#      never downloaded again on a cold start.
#   3. Run the app; the embedding model is loaded ONCE at startup and reused.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_CACHE=/app/.hf_cache/transformers

WORKDIR /app

# System deps required by lxml / psycopg / torch build tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the local embedding model so it is cached in the image.
# This is the ONLY download at build time; requests reuse the cached model.
COPY app/providers/embeddings.py /tmp/preload.py
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); print('embedding model cached')"

# Copy application code.
COPY . .

# Bind to the Railway domain target port (8080).
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
