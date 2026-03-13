from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant

from .schemas import ChunkRecord, DocumentRecord, RetrievedChunk


class QdrantChunkStore:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        collection_name: str,
        vector_size: int,
    ) -> None:
        if not url:
            raise ValueError("QDRANT_URL is required for vector storage.")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=url, api_key=api_key or None)

    def ensure_collection(self) -> None:
        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant.VectorParams(
                    size=self.vector_size,
                    distance=qdrant.Distance.COSINE,
                ),
            )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema=qdrant.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source_url",
            field_schema=qdrant.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="content_hash",
            field_schema=qdrant.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def fetch_doc_hashes(self) -> dict[str, str]:
        doc_hashes: dict[str, str] = {}
        next_offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["doc_id", "content_hash"],
                with_vectors=False,
                offset=next_offset,
                limit=256,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                content_hash = payload.get("content_hash")
                if isinstance(doc_id, str) and isinstance(content_hash, str) and doc_id:
                    doc_hashes[doc_id] = content_hash
            if next_offset is None:
                break

        return doc_hashes

    def delete_chunks_by_doc_ids(self, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant.FilterSelector(
                filter=qdrant.Filter(
                    must=[
                        qdrant.FieldCondition(
                            key="doc_id",
                            match=qdrant.MatchAny(any=doc_ids),
                        )
                    ]
                )
            ),
            wait=True,
        )
        return int(getattr(result, "count", 0) or 0)

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
        documents_by_id: dict[str, DocumentRecord],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors lengths must match.")
        if not chunks:
            return 0

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            document = documents_by_id.get(chunk.doc_id)
            if document is None:
                raise ValueError(f"Document metadata missing for doc_id={chunk.doc_id}")
            points.append(
                qdrant.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "source_url": chunk.source_url,
                        "heading_path": chunk.heading_path,
                        "token_count": chunk.token_count,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "title": document.title,
                        "doc_type": document.doc_type,
                        "section": document.section,
                        "content_hash": document.content_hash,
                        "last_seen_at": document.last_seen_at.isoformat(),
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search_chunks(self, query_vector: list[float], limit: int) -> list[RetrievedChunk]:
        if limit <= 0:
            return []

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(result, "points", result)

        hits: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id") or str(getattr(point, "id", ""))
            doc_id = payload.get("doc_id", "")
            text = payload.get("text", "")
            source_url = payload.get("source_url", "")
            if not chunk_id or not doc_id or not text or not source_url:
                continue

            hits.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    doc_id=str(doc_id),
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    text=str(text),
                    source_url=str(source_url),
                    title=payload.get("title"),
                    section=payload.get("section"),
                    heading_path=payload.get("heading_path"),
                )
            )
        return hits
