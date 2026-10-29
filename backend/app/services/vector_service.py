"""Qdrant vector database service for collection lifecycle, storage, and retrieval."""

from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.schemas.chunk import DocumentChunk


class VectorService:
    """Manages Qdrant vector database interactions."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        default_collection: Optional[str] = None,
        default_dimension: Optional[int] = None,
    ):
        self._custom_client = client
        self.default_collection = default_collection or settings.qdrant_collection
        self.default_dimension = default_dimension or settings.embedding_dimension

    @property
    def client(self) -> QdrantClient:
        """Return initialized QdrantClient singleton or custom client."""
        if self._custom_client is not None:
            return self._custom_client
        return get_qdrant_client()

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """Verify collection exists with matching dimension, or create it."""
        col = collection_name or self.default_collection
        dim = dimension or self.default_dimension

        client = self.client
        try:
            if not client.collection_exists(col):
                client.create_collection(
                    collection_name=col,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                return

            # Collection exists: verify dimension compatibility
            info = client.get_collection(col)
            # In qdrant-client, info.config.params.vectors can be VectorParams or dict
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, models.VectorParams):
                existing_dim = vectors_config.size
            elif isinstance(vectors_config, dict):
                # Named vectors or dict configuration
                existing_dim = next(iter(vectors_config.values())).size
            else:
                existing_dim = getattr(vectors_config, "size", None)

            if existing_dim is not None and existing_dim != dim:
                raise ValueError(
                    f"Incompatible vector dimension: Qdrant collection '{col}' "
                    f"has dimension {existing_dim}, but configured embedding dimension "
                    f"is {dim}. Explicit reindexing into a new collection is required."
                )
        except UnexpectedResponse as e:
            raise RuntimeError(
                f"Qdrant service responded with error during collection setup: {e}"
            ) from e
        except Exception as e:
            if "Incompatible vector dimension" in str(e):
                raise
            raise RuntimeError(
                f"Failed to ensure Qdrant collection '{col}': {e}"
            ) from e

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        filename: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Upsert document chunks and embeddings into Qdrant collection."""
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match "
                f"embedding count ({len(embeddings)})"
            )

        col = collection_name or self.default_collection
        self.ensure_collection(col)

        now_iso = datetime.now(timezone.utc).isoformat()
        points: list[models.PointStruct] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            payload = {
                "document_id": chunk.document_id,
                "filename": filename,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "section": chunk.section,
                "page_numbers": chunk.page_numbers,
                "character_count": chunk.character_count,
                "indexed_at": now_iso,
            }
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            self.client.upsert(collection_name=col, points=points)
            return len(points)
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(points)} vectors into collection '{col}': {e}"
            ) from e

    def search(
        self,
        query_vector: list[float],
        limit: int,
        collection_name: Optional[str] = None,
    ) -> list[dict]:
        """Perform cosine similarity vector search over the collection."""
        col = collection_name or self.default_collection

        # If collection does not exist, return empty list
        if not self.client.collection_exists(col):
            return []

        try:
            response = self.client.query_points(
                collection_name=col,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            hits: list[dict] = []
            for point in response.points:
                hits.append(
                    {
                        "id": str(point.id),
                        "score": float(point.score),
                        "payload": point.payload or {},
                    }
                )
            return hits
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute vector search in collection '{col}': {e}"
            ) from e

    def delete_document_vectors(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """Delete all vectors belonging to a specific document."""
        col = collection_name or self.default_collection
        if not self.client.collection_exists(col):
            return

        try:
            self.client.delete(
                collection_name=col,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete vectors for document '{document_id}' in '{col}': {e}"
            ) from e

    def count(
        self,
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> int:
        """Return total count of points, optionally filtered by document_id."""
        col = collection_name or self.default_collection
        if not self.client.collection_exists(col):
            return 0

        count_filter = None
        if document_id:
            count_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        try:
            res = self.client.count(collection_name=col, count_filter=count_filter)
            return res.count
        except Exception as e:
            raise RuntimeError(
                f"Failed to count points in collection '{col}': {e}"
            ) from e


# Singleton client instance
_qdrant_client_singleton: Optional[QdrantClient] = None
_vector_service_singleton: Optional[VectorService] = None


def get_qdrant_client() -> QdrantClient:
    """Return shared QdrantClient singleton to avoid multiple locks on disk storage."""
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        if settings.qdrant_url:
            _qdrant_client_singleton = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            qdrant_dir = settings.get_qdrant_path()
            qdrant_dir.mkdir(parents=True, exist_ok=True)
            _qdrant_client_singleton = QdrantClient(path=str(qdrant_dir))
    return _qdrant_client_singleton


def get_vector_service(client: Optional[QdrantClient] = None) -> VectorService:
    """Return VectorService instance."""
    global _vector_service_singleton
    if client is not None:
        return VectorService(client=client)
    if _vector_service_singleton is None:
        _vector_service_singleton = VectorService()
    return _vector_service_singleton


def reset_vector_service() -> None:
    """Reset vector service and client singletons (used in testing)."""
    global _qdrant_client_singleton, _vector_service_singleton
    if _qdrant_client_singleton is not None:
        try:
            _qdrant_client_singleton.close()
        except Exception:
            pass
        _qdrant_client_singleton = None
    _vector_service_singleton = None
