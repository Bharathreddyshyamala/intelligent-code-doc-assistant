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
    index_project,
)


router = APIRouter(
    tags=["Indexing"],
)


class IndexCodeRequest(BaseModel):
    project_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description=(
            "Project ID returned by an ingestion endpoint"
        ),
    )


class IndexCodeResponse(BaseModel):
    project_id: str
    file_count: int
    chunk_count: int
    embedding_count: int
    indexed_count: int
    status: str
    embedding_model: str
    collection_name: str


@router.post(
    "/index-code",
    response_model=IndexCodeResponse,
    status_code=status.HTTP_200_OK,
)
def index_code(
    request: IndexCodeRequest,
) -> IndexCodeResponse:
    """
    Chunk, embed, and index an already parsed project.
    """

    try:
        result = index_project(
            project_id=request.project_id,
        )

        return IndexCodeResponse(
            **result
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except EmbeddingServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except VectorStoreError as exc:
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
            detail=f"Project indexing failed: {exc}",
        ) from exc