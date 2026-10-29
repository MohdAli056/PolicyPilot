"""Service coordinating document chunk embedding and vector indexing."""

from typing import Optional

from app.schemas.index import IndexDocumentResponse
from app.services.bm25_service import BM25Service, bm25_service
from app.services.embedding_service import BaseEmbeddingProvider, get_embedding_provider
from app.services.storage_service import storage_service
from app.services.vector_service import VectorService, get_vector_service


class IndexingService:
    """Coordinates chunk retrieval, embedding generation, and vector indexing."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_service: Optional[VectorService] = None,
        bm25_retriever: Optional[BM25Service] = None,
    ):
        self._custom_embedding_provider = embedding_provider
        self._custom_vector_service = vector_service
        self._custom_bm25_service = bm25_retriever

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        return self._custom_embedding_provider or get_embedding_provider()

    @property
    def vector_service(self) -> VectorService:
        return self._custom_vector_service or get_vector_service()

    @property
    def bm25_retriever(self) -> BM25Service:
        return self._custom_bm25_service or bm25_service

    def index_document(self, document_id: str) -> IndexDocumentResponse:
        """Embed and index all chunks for a document into Qdrant.

        Reindexing is idempotent and deletion-safe:
        - Prior vectors for the document are pruned to prevent orphaned points
          if the document's chunk count decreased.
        - Chunk point IDs use deterministic UUIDv5 matching chunk_id.
        """
        # 1. Verify document exists
        doc = storage_service.get_document(document_id)
        if doc is None:
            raise KeyError(f"Document with ID '{document_id}' not found")

        # 2. Retrieve persisted chunks
        chunks = storage_service.get_chunks(document_id)
        if not chunks:
            return IndexDocumentResponse(
                document_id=document_id,
                indexed_chunks=0,
                status="empty",
                message="Document has no extractable chunks to index",
            )

        # 3. Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_provider.embed_texts(chunk_texts)

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding provider produced {len(embeddings)} vectors "
                f"for {len(chunks)} chunks"
            )

        # 4. Prune any prior vectors for this document to guarantee no stale orphans
        self.vector_service.delete_document_vectors(document_id=document_id)

        # 5. Upsert chunks with deterministic point IDs and payload metadata
        indexed_count = self.vector_service.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            filename=doc.filename,
        )

        # 6. Synchronize keyword search index
        self.bm25_retriever.index_document_chunks(
            document_id=document_id,
            filename=doc.filename,
            chunks=chunks,
        )

        return IndexDocumentResponse(
            document_id=document_id,
            indexed_chunks=indexed_count,
            status="completed",
            message=(
                f"Successfully indexed {indexed_count} chunks into "
                "vector and keyword stores"
            ),
        )


indexing_service = IndexingService()
