"""Question Answering API router for grounded policy responses with citations."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.qa import AskRequest, AskResponse
from app.services.llm_service import (
    LLMAuthenticationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ask", tags=["qa"])


@router.post(
    "",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a policy question with grounded evidence and citations",
)
async def ask_question(request: AskRequest) -> AskResponse:
    """Execute end-to-end RAG question answering over indexed policy documents.

    Retrieves candidates using hybrid vector+BM25 search, reranks with cross-encoder,
    synthesizes an answer using bounded evidence, validates all citations on the server,
    and returns explicit abstention if evidence is insufficient.
    """
    try:
        response = await asyncio.to_thread(
            rag_service.answer_question,
            request=request,
        )
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except LLMAuthenticationError as exc:
        logger.error("LLM authentication failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "LLM provider is not configured. Please ensure GEMINI_API_KEY "
                "is set in the environment."
            ),
        ) from exc
    except LLMRateLimitError as exc:
        logger.warning("LLM rate limit reached: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="LLM provider rate limit exceeded. Please try again shortly.",
        ) from exc
    except LLMTimeoutError as exc:
        logger.warning("LLM request timed out: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM generation timed out. Please try again.",
        ) from exc
    except LLMProviderError as exc:
        logger.error("LLM provider error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM generation provider error. Please check backend logs.",
        ) from exc
    except LLMError as exc:
        logger.error("General LLM error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in RAG answer pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the answer.",
        ) from exc
