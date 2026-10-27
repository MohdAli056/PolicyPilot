"""Document chunking service for policy documents.

Implements deterministic, paragraph- and heading-aware text chunking with
configurable target size, overlap, section detection, and one-based page attribution.
"""

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.schemas.chunk import DocumentChunk
from app.schemas.document import PageExtraction

# Regex patterns for heading and section detection
RE_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
RE_NUMBERED_HEADING = re.compile(
    r"^(\d+(\.\d+)*\.?)\s+([A-Z][A-Za-z0-9\s,\-–—/()]{2,80})$"
)
RE_KEYWORD_HEADING = re.compile(
    r"^(Section|Chapter|Article|Policy|Part)\s+([0-9A-Za-z]+)[:\s\-\.]+(.+)$",
    re.IGNORECASE,
)
RE_ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9\s,\-–—/()]{4,70}$")


@dataclass
class TextSegment:
    """An atomic text unit with its source page number and active section."""

    text: str
    page_number: int
    section: Optional[str]


def detect_heading(line: str) -> Optional[str]:
    """Detect whether a line of text represents a section or heading.

    Heuristic detection recognizes:
      1. Markdown headings (# Heading)
      2. Numbered clauses (e.g., '1. Purpose', '2.1 Eligibility')
      3. Section keywords (e.g., 'Section 1: Leave Policy', 'Article 4')
      4. Short standalone lines in ALL-CAPS (e.g., 'CODE OF CONDUCT')

    Limitations:
      Heuristics rely on structural markers, prefixes, and casing.
      Headings distinguished solely by font size/weight in PDFs are not
      guaranteed to be detected.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None

    # Check Markdown heading
    m_md = RE_MD_HEADING.match(stripped)
    if m_md:
        return m_md.group(1).strip()

    # Check Section/Chapter/Article keyword heading
    m_kw = RE_KEYWORD_HEADING.match(stripped)
    if m_kw:
        return stripped

    # Check numbered heading (e.g., '1. Purpose' or '2.1 Scope')
    m_num = RE_NUMBERED_HEADING.match(stripped)
    if m_num:
        return stripped

    # Check all-caps standalone line (must have at least one letter and no lowercase)
    if (
        RE_ALL_CAPS_HEADING.match(stripped)
        and any(c.isalpha() for c in stripped)
        and not any(c.islower() for c in stripped)
    ):
        # Exclude common false positives like dates or page numbers
        if len(stripped.split()) >= 2:
            return stripped

    return None


def generate_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Generate a deterministic, reproducible unique identifier for a chunk.

    Derived from document_id, chunk_index, and SHA-256 hash of chunk text.
    Uses uuid5 with NAMESPACE_OID for consistent UUID formatting.
    """
    text_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    content_key = f"{document_id}:{chunk_index}:{text_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, content_key))


def _split_into_atomic_segments(
    pages: list[PageExtraction],
    target_size: int,
) -> list[TextSegment]:
    """Break extracted pages into atomic text segments tagged with page and section.

    Paragraphs are prioritized; oversized blocks are safely split by sentences
    or character bounds to ensure no single segment exceeds target_size.
    """
    segments: list[TextSegment] = []
    current_section: Optional[str] = None

    for page in pages:
        raw_page_text = page.text or ""
        if not raw_page_text.strip():
            continue

        # Split page into paragraph blocks
        raw_paragraphs = re.split(r"\n\s*\n", raw_page_text)

        for raw_para in raw_paragraphs:
            cleaned_para = raw_para.strip()
            if not cleaned_para:
                continue

            # Check if paragraph begins with or is a heading
            lines = [line.strip() for line in cleaned_para.split("\n") if line.strip()]
            if lines:
                detected = detect_heading(lines[0])
                if detected:
                    current_section = detected

            # If paragraph fits within target_size, add directly
            if len(cleaned_para) <= target_size:
                segments.append(
                    TextSegment(
                        text=cleaned_para,
                        page_number=page.page_number,
                        section=current_section,
                    )
                )
            else:
                # Subdivide oversized paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", cleaned_para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(sentence) <= target_size:
                        segments.append(
                            TextSegment(
                                text=sentence,
                                page_number=page.page_number,
                                section=current_section,
                            )
                        )
                    else:
                        # Break oversized sentence by words or character slices
                        words = sentence.split(" ")
                        current_word_block = ""
                        for word in words:
                            # If a single word is larger than target_size, slice it
                            while len(word) > target_size:
                                if current_word_block:
                                    segments.append(
                                        TextSegment(
                                            text=current_word_block,
                                            page_number=page.page_number,
                                            section=current_section,
                                        )
                                    )
                                    current_word_block = ""
                                segments.append(
                                    TextSegment(
                                        text=word[:target_size],
                                        page_number=page.page_number,
                                        section=current_section,
                                    )
                                )
                                word = word[target_size:]

                            candidate = (
                                f"{current_word_block} {word}".strip()
                                if current_word_block
                                else word
                            )
                            if len(candidate) <= target_size:
                                current_word_block = candidate
                            else:
                                if current_word_block:
                                    segments.append(
                                        TextSegment(
                                            text=current_word_block,
                                            page_number=page.page_number,
                                            section=current_section,
                                        )
                                    )
                                current_word_block = word

                        if current_word_block:
                            segments.append(
                                TextSegment(
                                    text=current_word_block,
                                    page_number=page.page_number,
                                    section=current_section,
                                )
                            )

    return segments


def chunk_document(
    document_id: str,
    pages: list[PageExtraction],
    target_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[DocumentChunk]:
    """Chunk a document's extracted pages into deterministic, overlapping chunks.

    Args:
        document_id: UUID of the parent document.
        pages: List of extracted pages with 1-based page numbers.
        target_chunk_size: Target size in characters (defaults to
            settings.target_chunk_size).
        chunk_overlap: Overlap in characters (defaults to settings.chunk_overlap).

    Returns:
        List of DocumentChunk instances in sequential chunk_index order.

    Raises:
        ValueError: If target_chunk_size <= 0 or chunk_overlap < 0 or
                    chunk_overlap >= target_chunk_size.
    """
    target_size = (
        target_chunk_size
        if target_chunk_size is not None
        else settings.target_chunk_size
    )
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if target_size <= 0:
        raise ValueError(f"target_chunk_size must be positive, got {target_size}")
    if overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {overlap}")
    if overlap >= target_size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be strictly less than "
            f"target_chunk_size ({target_size})"
        )

    # Convert pages into atomic segments <= target_size
    segments = _split_into_atomic_segments(pages, target_size)
    if not segments:
        return []

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    start_idx = 0
    num_segments = len(segments)

    while start_idx < num_segments:
        current_segments: list[TextSegment] = []
        current_len = 0
        end_idx = start_idx

        while end_idx < num_segments:
            seg = segments[end_idx]
            add_len = len(seg.text) + (
                2 if current_segments else 0
            )  # account for "\n\n"

            # Always include at least one segment
            if current_segments and (current_len + add_len > target_size):
                break

            current_segments.append(seg)
            current_len += add_len
            end_idx += 1

        # Build chunk text and metadata
        chunk_text = "\n\n".join(s.text for s in current_segments).strip()
        contributing_pages = sorted(list({s.page_number for s in current_segments}))
        chunk_section = current_segments[0].section

        chunk_id = generate_chunk_id(document_id, chunk_index, chunk_text)

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                section=chunk_section,
                page_numbers=contributing_pages,
                character_count=len(chunk_text),
            )
        )
        chunk_index += 1

        if end_idx >= num_segments:
            break

        # Calculate next start index based on overlap
        if overlap > 0:
            overlap_accum = 0
            back_idx = end_idx - 1
            while back_idx > start_idx:
                seg_len = len(segments[back_idx].text)
                if overlap_accum + seg_len <= overlap:
                    overlap_accum += seg_len
                    back_idx -= 1
                else:
                    break
            # Advance to at least start_idx + 1 to prevent infinite loops
            start_idx = max(start_idx + 1, back_idx + 1)
        else:
            start_idx = end_idx

    return chunks
