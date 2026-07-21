from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.file_scanner import (
    get_ast_path,
    get_chunks_path,
    get_source_directory,
    read_metadata,
    update_metadata,
    validate_project_id,
)
from services.indexer import index_project


router = APIRouter(
    tags=["Indexing"],
)


class IndexCodeRequest(BaseModel):
    project_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="Project ID returned by an ingestion API",
    )


class IndexCodeResponse(BaseModel):
    project_id: str
    file_count: int
    chunk_count: int
    status: str


@router.post(
    "/index-code",
    response_model=IndexCodeResponse,
    status_code=status.HTTP_200_OK,
)
def index_code(
    request: IndexCodeRequest,
) -> IndexCodeResponse:
    """
    Start chunking, embedding, and ChromaDB indexing
    for an already parsed project.
    """

    try:
        validate_project_id(request.project_id)

        metadata = read_metadata(
            request.project_id
        )

        current_status = metadata.get("status")

        if current_status not in {
            "parsed",
            "chunked",
            "indexed",
        }:
            raise ValueError(
                "The project must be parsed before indexing. "
                "Call /parse-code first."
            )

        # Avoid creating duplicate vectors when the user
        # presses the Index Project button again.
        if current_status == "indexed":
            return IndexCodeResponse(
                project_id=request.project_id,
                file_count=metadata.get("file_count", 0),
                chunk_count=metadata.get("chunk_count", 0),
                status="indexed",
            )

        source_directory = get_source_directory(
            request.project_id
        )

        ast_path = get_ast_path(
            request.project_id
        )

        chunks_output_path = get_chunks_path(
            request.project_id
        )

        if not source_directory.exists():
            raise FileNotFoundError(
                "The project source folder was not found."
            )

        if not ast_path.exists():
            raise FileNotFoundError(
                "ast.json was not found. "
                "Call /parse-code before /index-code."
            )

        update_metadata(
            request.project_id,
            status="indexing",
        )

        result = index_project(
            project_id=request.project_id,
            source_directory=source_directory,
            ast_path=ast_path,
            chunks_output_path=chunks_output_path,
            metadata=metadata,
        )

        chunk_count = result.get(
            "chunk_count",
            0,
        )

        update_metadata(
            request.project_id,
            status="indexed",
            chunk_count=chunk_count,
            indexing_error=None,
        )

        return IndexCodeResponse(
            project_id=request.project_id,
            file_count=metadata.get("file_count", 0),
            chunk_count=chunk_count,
            status="indexed",
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        try:
            update_metadata(
                request.project_id,
                status="indexing_failed",
                indexing_error=str(error),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project indexing failed: {error}",
        ) from error