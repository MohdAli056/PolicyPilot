"""Pydantic schemas for Grounded Question Answering, Citations, and Abstention."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CitationItem(BaseModel):
    """A verified citation pointing to an exact evidence chunk and page numbers.

    Citations are validated and resolved strictly on the server against the actual
    retrieved passages, preventing hallucinated page numbers or chunk IDs.
    """

    source_id: str = Field(
        ...,
        description="Stable citation identifier referenced in the answer (e.g. 'S1')",
    )
    document_id: str = Field(
        ..., description="Unique UUID of the cited policy document"
    )
    filename: str = Field(..., description="Original filename of the policy document")
    chunk_id: str = Field(
        ..., description="Deterministic unique identifier of the supporting chunk"
    )
    page_numbers: list[int] = Field(
        ..., description="One-based source page numbers contributing the cited text"
    )
    section: Optional[str] = Field(
        None, description="Heading or policy section title where available"
    )
    excerpt: str = Field(
        ...,
        description="Verbatim excerpt or passage reference from the supporting chunk",
    )


class SourceItem(BaseModel):
    """Full metadata for an evidence passage provided to the LLM context."""

    source_id: str = Field(
        ..., description="Identifier assigned to the passage (e.g. 'S1')"
    )
    document_id: str = Field(..., description="Unique UUID of the source document")
    filename: str = Field(..., description="Original filename of the policy document")
    chunk_id: str = Field(..., description="Deterministic unique identifier of chunk")
    chunk_index: int = Field(..., description="Zero-based index of chunk in document")
    page_numbers: list[int] = Field(
        ..., description="One-based source page numbers contributing text"
    )
    section: Optional[str] = Field(None, description="Section heading or None")
    text: str = Field(..., description="Full text of the evidence passage")
    rerank_score: float = Field(
        ..., description="Cross-encoder relevance score for this passage"
    )
    rerank_rank: int = Field(
        ..., description="Final rank of the passage after cross-encoder scoring"
    )
    rrf_score: float = Field(..., description="Initial hybrid retrieval RRF score")
    retrievers: list[str] = Field(
        ...,
        description="Retrieval channels that surfaced this chunk ('vector', 'bm25')",
    )


class ClaimItem(BaseModel):
    """An individual factual claim or sentence with its supporting citations."""

    claim_text: str = Field(
        ..., description="Sentence or statement segment from the answer"
    )
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Source identifiers cited for this specific claim (e.g. ['S1'])",
    )
    is_grounded: bool = Field(
        ...,
        description="True if claim has verified lexical support in cited evidence",
    )
    grounding_notes: Optional[str] = Field(
        None, description="Diagnostic notes or warnings regarding grounding"
    )


class AskRequest(BaseModel):
    """Request payload for answering a policy question."""

    question: str = Field(
        ..., description="Natural language question about company policies"
    )
    limit: Optional[int] = Field(
        None,
        description="Maximum evidence passages to provide to LLM",
    )
    candidate_limit: Optional[int] = Field(
        None,
        description="Candidate chunks retrieved for cross-encoder reranking",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace only")
        if len(stripped) < 2:
            raise ValueError("Question must be at least 2 characters in length")
        if len(stripped) > 500:
            raise ValueError(
                "Question exceeds maximum allowed length of 500 characters"
            )
        return stripped

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 20):
            raise ValueError(f"Evidence limit must be between 1 and 20, got {v}")
        return v

    @field_validator("candidate_limit")
    @classmethod
    def validate_candidate_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 50):
            raise ValueError(f"Candidate limit must be between 1 and 50, got {v}")
        return v


class AskResponse(BaseModel):
    """Structured response for grounded policy question answering."""

    question: str = Field(..., description="The user question that was answered")
    answer: str = Field(
        ..., description="Grounded answer text with inline citation tags"
    )
    is_abstention: bool = Field(
        ...,
        description="True if system abstained due to insufficient evidence",
    )
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="Validated citations directly backing factual claims in the answer",
    )
    claims: list[ClaimItem] = Field(
        default_factory=list,
        description="Claim-level grounding breakdown with supporting citation IDs",
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="All evidence passages supplied to the LLM within context budget",
    )
    model: str = Field(
        ..., description="Name or identifier of the generation model used"
    )
    timing_ms: Optional[dict[str, float]] = Field(
        None,
        description="Execution latency breakdown in milliseconds",
    )
