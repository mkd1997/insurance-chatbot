## Insurance Chatbot

Step 1 scaffold is in place:

- `backend/app/main.py` provides a minimal FastAPI app with `GET /health`.
- `backend/app/config.py` loads `.env` values via `pydantic-settings`.
- `streamlit_app/app.py` provides a Streamlit shell to test backend health.

Step 2 API contracts are now scaffolded:

- `POST /ingest` accepts `{"mode":"incremental|full","seed_url": "...optional..."}`.
- `GET /ingest/status` returns typed ingestion status metadata and counters.
- `POST /chat` accepts `{"question":"...","top_k":6}` and currently returns a strict refusal stub.

Step 3 crawler scaffold is now added:

- `backend/app/crawler.py` implements Playwright-based policy link discovery.
- Discovery is constrained to same-domain, policy-path URLs and classifies HTML vs PDF links.
- Deterministic link filtering/classification tests are in `tests/test_crawler.py`.

Step 4 extraction scaffold is now added:

- `backend/app/extractor.py` parses HTML and PDF policy content into normalized text + metadata.
- HTML extraction captures `title` and top section heading when present.
- PDF extraction normalizes per-page text and preserves source metadata.
- Unit tests are in `tests/test_extractor.py`.

Step 5 chunking scaffold is now added:

- `backend/app/chunker.py` generates token-aware semantic chunks from extracted documents.
- Deterministic IDs:
  - `doc_id` from source URL + doc type
  - `chunk_id` from doc ID + chunk index + chunk text
- Chunk metadata includes `heading_path`, `token_count`, `chunk_index`, and source URL.
- Unit tests are in `tests/test_chunker.py`.

Step 6 embedding and vector-store scaffold is now added:

- `backend/app/embeddings.py` adds OpenAI embedding generation (`OpenAIEmbeddingClient`).
- `backend/app/vector_store.py` adds Qdrant collection setup + chunk upsert support.
- Collection setup creates payload indexes for `doc_id` and `source_url`.
- Unit tests are in `tests/test_embeddings.py` and `tests/test_vector_store.py`.

Step 7 incremental ingestion is now added:

- `backend/app/ingestion.py` orchestrates crawl -> extract -> hash compare -> chunk -> embed -> upsert.
- Hash-based refresh behavior:
  - unchanged docs are skipped,
  - changed docs have old chunks deleted then replaced,
  - stale docs (missing from latest crawl) are deleted from the vector store.
- `POST /ingest` now runs incremental ingestion and updates status counters.
- Unit regression coverage for changed/unchanged/stale document handling is in `tests/test_ingestion.py`.

Step 8 retrieval + strict refusal is now added:

- `backend/app/qa.py` implements retrieval-driven answering with strict refusal thresholding.
- `backend/app/vector_store.py` now supports scored nearest-neighbor retrieval (`search_chunks`).
- `backend/app/llm.py` adds OpenAI chat answer generation constrained to retrieved context.
- `POST /chat` now runs the QA flow and refuses when retrieval confidence is insufficient.
- Unit coverage is in `tests/test_qa.py` and extended `tests/test_vector_store.py`.

Step 9 citation enforcement is now added:

- Non-refusal answers must include valid citations (source URL + excerpt).
- Citation output is deduplicated by source/section to avoid repeated references.
- If an answer is generated but valid citations cannot be built, the response is forced to strict refusal.

Step 10 Streamlit UI is now added:

- `streamlit_app/app.py` includes admin controls for health check, re-ingest trigger, and ingestion status refresh.
- Chat UI supports question submission, configurable `top_k`, response history, and citation rendering.
- Non-refusal responses show citations and retrieval scores for traceability.
- Admin actions now require entering `ADMIN_PASSWORD` for each button action.

Step 11 hardening and deployment notes are now added:

- Basic structured logging is enabled in backend startup.
- Retry logic is applied to outbound network calls (OpenAI embeddings/chat and document fetches).
- README now includes a deployment configuration and smoke checklist.

### Environment setup

Create a `.env` file in the project root and fill at least:

- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` (if required)
- `ADMIN_PASSWORD`

### Streamlit Community Cloud deployment notes

- Set the same values in Streamlit secrets/environment configuration.
- Ensure backend URL is reachable from Streamlit app:
  - `STREAMLIT_BACKEND_URL=<your-fastapi-url>`
- Keep `ADMIN_PASSWORD` secret only in deployment settings (never commit it).

### Run backend

```bash
conda run -n insurance-chatbot uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Streamlit shell

```bash
conda run -n insurance-chatbot streamlit run streamlit_app/app.py
```

### Smoke checklist

1. Start backend and Streamlit successfully.
2. Health check works from Streamlit admin panel (with admin password).
3. Run incremental re-ingest and confirm `completed` with counters > 0.
4. Ask a known in-scope policy question and verify:
   - non-refusal answer,
   - citations shown.
5. Ask an out-of-scope question and verify strict refusal:
   - `I don't know based on the provided policy documents.`
