from datetime import datetime, timezone
import logging
from uuid import uuid4

from fastapi import FastAPI

from .config import get_settings
from .embeddings import OpenAIEmbeddingClient
from .ingestion import IngestionService
from .llm import OpenAIChatClient
from .logging_utils import configure_logging
from .qa import QAService, REFUSAL_TEXT
from .schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    IngestStatusResponse,
    RetrievalDebug,
)
from .vector_store import QdrantChunkStore

settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
ingest_status = IngestStatusResponse()


def _build_ingestion_service() -> IngestionService:
    store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
    )
    embedding_client = OpenAIEmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    return IngestionService(store=store, embedding_client=embedding_client)


def _build_qa_service() -> QAService:
    store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
    )
    embedding_client = OpenAIEmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    llm_client = OpenAIChatClient(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
    )
    return QAService(
        embedding_client=embedding_client,
        store=store,
        llm_client=llm_client,
        score_threshold=settings.retrieval_score_threshold,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    run_id = f"ingest-{uuid4().hex[:10]}"
    logger.info("Ingest requested: run_id=%s mode=%s", run_id, request.mode)
    ingest_status.status = "running"
    ingest_status.mode = request.mode
    ingest_status.run_id = run_id
    ingest_status.message = "Ingestion in progress."
    ingest_status.last_run_at = datetime.now(timezone.utc)
    ingest_status.counters.discovered_urls = 0
    ingest_status.counters.processed_docs = 0
    ingest_status.counters.upserted_chunks = 0
    ingest_status.counters.deleted_chunks = 0

    try:
        if request.mode != "incremental":
            raise ValueError("Only incremental mode is implemented in this version.")

        seed_url = request.seed_url or settings.policy_seed_url
        service = _build_ingestion_service()
        counters = service.run_incremental(seed_url=seed_url)

        ingest_status.counters = counters
        ingest_status.status = "completed"
        ingest_status.message = "Incremental ingestion completed."
        logger.info(
            "Ingest completed: run_id=%s upserted=%d deleted=%d",
            run_id,
            counters.upserted_chunks,
            counters.deleted_chunks,
        )
        return IngestResponse(
            accepted=True,
            status="completed",
            run_id=run_id,
            message=ingest_status.message,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed: run_id=%s", run_id)
        ingest_status.status = "failed"
        ingest_status.message = str(exc)
        return IngestResponse(
            accepted=False,
            status="failed",
            run_id=run_id,
            message=ingest_status.message,
        )


@app.get("/ingest/status", response_model=IngestStatusResponse)
def get_ingest_status() -> IngestStatusResponse:
    return ingest_status


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        service = _build_qa_service()
        response = service.answer_question(request.question, request.top_k)
        logger.info("Chat handled: refusal=%s top_k=%d", response.is_refusal, request.top_k)
        return response
    except Exception:
        logger.exception("Chat handling failed")
        return ChatResponse(
            answer=REFUSAL_TEXT,
            is_refusal=True,
            citations=[],
            retrieval_debug=RetrievalDebug(top_k_scores=[]),
        )
