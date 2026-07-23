from typing import List

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import BaseModel, Field

from services.embedding_service import (
    EmbeddingServiceError,
)
from services.indexer import (
    VectorStoreError,
)
from services.ollama_service import (
    OllamaServiceError,
)
from services.rag_service import (
    answer_code_question,
)
from services.retrieval_service import (
    RetrievalServiceError,
)


router = APIRouter(
    tags=["RAG"],
)


class AskCodeRequest(BaseModel):
    project_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description=(
            "Project ID returned by an ingestion endpoint"
        ),
    )

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description=(
            "Question about the indexed source code"
        ),
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description=(
            "Number of code chunks to retrieve"
        ),
    )


class SourceReference(BaseModel):
    source_number: int
    chunk_id: str
    file_path: str
    symbol: str
    source_type: str
    start_line: int
    end_line: int
    distance: float
    excerpt: str


class AskCodeResponse(BaseModel):
    project_id: str
    question: str
    answer: str
    chat_model: str
    retrieved_chunk_count: int
    used_chunk_count: int
    sources: List[SourceReference]


@router.post(
    "/ask-code",
    response_model=AskCodeResponse,
    status_code=status.HTTP_200_OK,
)
def ask_code(
    request: AskCodeRequest,
) -> AskCodeResponse:
    """
    Answer a question about an indexed project using RAG.
    """

    try:
        result = answer_code_question(
            project_id=request.project_id,
            question=request.question,
            top_k=request.top_k,
        )

        return AskCodeResponse(
            **result
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except (
        EmbeddingServiceError,
        OllamaServiceError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except (
        RetrievalServiceError,
        VectorStoreError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Ask Code request failed: "
                f"{exc}"
            ),
        ) from exc