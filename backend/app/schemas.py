from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(BaseModel):
    doc_id: str
    source_url: str
    title: str
    doc_type: Literal["html", "pdf"]
    section: str | None = None
    content_hash: str | None = None
    last_seen_at: datetime = Field(default_factory=utc_now)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    source_url: str
    heading_path: str | None = None
    token_count: int
    chunk_index: int


class Citation(BaseModel):
    source_url: str
    title: str | None = None
    section: str | None = None
    excerpt: str


class RetrievalDebug(BaseModel):
    top_k_scores: list[float] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    source_url: str
    title: str | None = None
    section: str | None = None
    heading_path: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    is_refusal: bool
    citations: list[Citation] = Field(default_factory=list)
    retrieval_debug: RetrievalDebug | None = None


class IngestRequest(BaseModel):
    mode: Literal["incremental", "full"] = "incremental"
    seed_url: str | None = None


class IngestCounters(BaseModel):
    discovered_urls: int = 0
    processed_docs: int = 0
    upserted_chunks: int = 0
    deleted_chunks: int = 0


class IngestStatusResponse(BaseModel):
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    mode: Literal["incremental", "full"] = "incremental"
    run_id: str | None = None
    message: str | None = None
    last_run_at: datetime | None = None
    counters: IngestCounters = Field(default_factory=IngestCounters)


class IngestResponse(BaseModel):
    accepted: bool
    status: Literal["queued", "running", "completed", "failed"]
    run_id: str
    message: str
