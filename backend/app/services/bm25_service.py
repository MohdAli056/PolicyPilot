"""BM25 Keyword Retrieval Service.

Implements Okapi BM25 with Lucene-adjusted IDF for exact keyword retrieval
across persisted document chunks. Synchronizes in-memory search index with
disk-persisted chunks created by the ingestion pipeline.
"""

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings
from app.schemas.chunk import DocumentChunk
from app.schemas.hybrid_search import HybridSearchResultItem


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric and identifier tokens.

    Tokenization Strategy:
    1. Case folding: Converts all input to lowercase.
    2. Identifiers & Numbers: Extracts word and numeric sequences, preserving
       compound policy identifiers (e.g. 'HR-2024', '1.1', 'SEC-101', 'ISO-9001').
    3. Sub-token emission: For compound tokens containing hyphens or dots, also
       emits their constituent alphanumeric components so queries for either
       the composite ('HR-2024') or the subpart ('2024') match.
    4. Punctuation handling: Trailing and surrounding punctuation ($, %, commas,
       quotes, parentheses, brackets, colons) are cleanly stripped.

    Limitations:
    - Pure non-Latin scripts or languages without whitespace/boundary delimiters
      require specialized segmentation not included in this English-focused pipeline.
    - Stemming/lemmatization is intentionally omitted to prioritize exact policy
      term and code preservation without stemming collisions.
    """
    if not text:
        return []
    tokens: list[str] = []
    raw = re.findall(r"[a-zA-Z0-9]+(?:[.\-_][a-zA-Z0-9]+)*", text.lower())
    for token in raw:
        tokens.append(token)
        sub = re.findall(r"[a-zA-Z0-9]+", token)
        if len(sub) > 1:
            tokens.extend(sub)
    return tokens


@dataclass
class IndexedChunkRecord:
    """In-memory representation of an indexed chunk for BM25 retrieval."""

    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    text: str
    section: Optional[str]
    page_numbers: list[int]
    character_count: int
    token_counts: dict[str, int]
    length: int


class BM25Service:
    """Thread-safe BM25 retrieval service synchronized with persisted chunks."""

    def __init__(
        self,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ):
        self._k1 = k1 if k1 is not None else settings.bm25_k1
        self._b = b if b is not None else settings.bm25_b
        self._lock = threading.RLock()

        # Indexed documents: document_id -> list[IndexedChunkRecord]
        self._documents: dict[str, list[IndexedChunkRecord]] = {}

        # Cached index state for fast retrieval
        self._all_chunks: list[IndexedChunkRecord] = []
        self._chunk_by_id: dict[str, IndexedChunkRecord] = {}
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    @property
    def total_chunks(self) -> int:
        """Return the total number of indexed chunks across all documents."""
        with self._lock:
            return len(self._all_chunks)

    @property
    def total_documents(self) -> int:
        """Return the number of unique indexed documents."""
        with self._lock:
            return len(self._documents)

    def _rebuild_index_stats(self) -> None:
        """Recalculate document frequencies, IDF, and avg document length.

        Must be called while holding self._lock.
        """
        all_chunks: list[IndexedChunkRecord] = []
        chunk_by_id: dict[str, IndexedChunkRecord] = {}
        doc_freqs: dict[str, int] = Counter()
        total_length = 0

        for chunk_list in self._documents.values():
            for chunk in chunk_list:
                all_chunks.append(chunk)
                chunk_by_id[chunk.chunk_id] = chunk
                total_length += chunk.length
                for term in chunk.token_counts.keys():
                    doc_freqs[term] += 1

        n_chunks = len(all_chunks)
        avgdl = (total_length / n_chunks) if n_chunks > 0 else 0.0

        # Lucene-adjusted BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        # Guaranteed to be strictly positive for all 1 <= df <= N
        idf: dict[str, float] = {}
        if n_chunks > 0:
            for term, df in doc_freqs.items():
                idf[term] = math.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5))

        self._all_chunks = all_chunks
        self._chunk_by_id = chunk_by_id
        self._doc_freqs = dict(doc_freqs)
        self._idf = idf
        self._avgdl = avgdl

    def index_document_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[DocumentChunk],
    ) -> int:
        """Atomically index or replace chunks for a given document.

        Guarantees that re-indexing an existing document replaces prior chunks
        without leaving stale orphans or duplicates.
        """
        records: list[IndexedChunkRecord] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            token_counts = dict(Counter(tokens))
            records.append(
                IndexedChunkRecord(
                    document_id=document_id,
                    filename=filename,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    section=chunk.section,
                    page_numbers=chunk.page_numbers,
                    character_count=chunk.character_count,
                    token_counts=token_counts,
                    length=len(tokens),
                )
            )

        with self._lock:
            self._documents[document_id] = records
            self._rebuild_index_stats()
            return len(records)

    def delete_document(self, document_id: str) -> None:
        """Remove a document from the keyword index if present."""
        with self._lock:
            if document_id in self._documents:
                del self._documents[document_id]
                self._rebuild_index_stats()

    def sync_from_storage(self, storage_service) -> int:
        """Synchronize in-memory index with all persisted chunks on disk.

        Scans the chunks storage directory and builds/rebuilds the index.
        Runs at startup or on explicit resynchronization.
        """
        chunks_dir = Path(storage_service.chunks_dir)
        if not chunks_dir.exists():
            with self._lock:
                self._documents.clear()
                self._rebuild_index_stats()
            return 0

        new_docs: dict[str, list[IndexedChunkRecord]] = {}
        for chunk_file in chunks_dir.glob("*.json"):
            doc_id = chunk_file.stem
            if not storage_service.validate_document_id(doc_id):
                continue
            chunks = storage_service.get_chunks(doc_id)
            if not chunks:
                continue
            doc = storage_service.get_document(doc_id)
            filename = doc.filename if doc else f"{doc_id}.pdf"

            records: list[IndexedChunkRecord] = []
            for chunk in chunks:
                tokens = tokenize(chunk.text)
                token_counts = dict(Counter(tokens))
                records.append(
                    IndexedChunkRecord(
                        document_id=doc_id,
                        filename=filename,
                        chunk_id=chunk.chunk_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        section=chunk.section,
                        page_numbers=chunk.page_numbers,
                        character_count=chunk.character_count,
                        token_counts=token_counts,
                        length=len(tokens),
                    )
                )
            new_docs[doc_id] = records

        with self._lock:
            self._documents = new_docs
            self._rebuild_index_stats()
            return len(self._all_chunks)

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> list[HybridSearchResultItem]:
        """Perform BM25 keyword retrieval over indexed document chunks.

        Returns ranked list of matching chunks. Returns empty list if:
        - Query is empty or has no searchable tokens
        - Corpus is empty
        - No chunks match the query (never fabricates non-matching results)
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        query_tokens = tokenize(clean_query)
        if not query_tokens:
            return []

        eff_limit = limit if limit is not None else settings.default_search_limit
        if eff_limit <= 0:
            raise ValueError(f"Search limit must be positive, got {eff_limit}")
        if eff_limit > settings.max_search_limit:
            eff_limit = settings.max_search_limit

        with self._lock:
            if not self._all_chunks:
                return []

            # Filter query terms present in the vocabulary
            matched_terms = [t for t in query_tokens if t in self._idf]
            if not matched_terms:
                return []

            scored_candidates: list[tuple[float, IndexedChunkRecord]] = []

            for chunk in self._all_chunks:
                score = 0.0
                for term in matched_terms:
                    tf = chunk.token_counts.get(term, 0)
                    if tf > 0:
                        idf = self._idf[term]
                        # BM25 term score calculation
                        numerator = tf * (self._k1 + 1.0)
                        denom = tf + self._k1 * (
                            1.0 - self._b + self._b * (chunk.length / self._avgdl)
                        )
                        score += idf * (numerator / denom)

                if score > 0.0:
                    scored_candidates.append((score, chunk))

            if not scored_candidates:
                return []

            # Deterministic sorting: highest score first,
            # stable secondary sort by chunk_id
            scored_candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))

            results: list[HybridSearchResultItem] = []
            for rank_idx, (score, chunk) in enumerate(
                scored_candidates[:eff_limit], start=1
            ):
                results.append(
                    HybridSearchResultItem(
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        chunk_id=chunk.chunk_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        section=chunk.section,
                        page_numbers=chunk.page_numbers,
                        character_count=chunk.character_count,
                        vector_rank=None,
                        keyword_rank=rank_idx,
                        rrf_score=round(score, 4),
                        retrievers=["bm25"],
                    )
                )

            return results


bm25_service = BM25Service()
