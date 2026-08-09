from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services.doc_generator import (
    DocumentationGenerationError,
    generate_project_documentation,
    get_generated_documentation_path,
)
from services.file_scanner import validate_project_id


router = APIRouter(
    tags=["Documentation"],
)


class GenerateDocsRequest(BaseModel):
    project_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="Project ID returned by an ingestion endpoint",
    )
    project_title: Optional[str] = Field(
        default=None,
        max_length=150,
    )
    max_files: int = Field(
        default=30,
        ge=1,
        le=100,
    )
    overwrite: bool = True


class GenerateDocsResponse(BaseModel):
    project_id: str
    project_title: str
    status: str
    output_format: str
    documented_file_count: int
    skipped_file_count: int
    output_path: str
    preview_url: str
    download_url: str


@router.post(
    "/generate-docs",
    response_model=GenerateDocsResponse,
    status_code=status.HTTP_200_OK,
)
def generate_docs(
    request: GenerateDocsRequest,
) -> GenerateDocsResponse:
    """Generate a project documentation PDF."""
    try:
        result = generate_project_documentation(
            project_id=request.project_id,
            project_title=request.project_title,
            max_files=request.max_files,
            overwrite=request.overwrite,
        )
        return GenerateDocsResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    except DocumentationGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected documentation-generation error: {}".format(error),
        ) from error


def _validated_pdf_path(project_id: str):
    """Validate the project ID and return an existing PDF path."""
    try:
        validate_project_id(project_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    pdf_path = get_generated_documentation_path(project_id)

    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Generated PDF documentation was not found. "
                "Call POST /generate-docs first."
            ),
        )

    return pdf_path


@router.get(
    "/generated-docs/{project_id}",
    response_class=FileResponse,
)
def preview_generated_docs(
    project_id: str,
) -> FileResponse:
    """Open the generated PDF in the browser when supported."""
    pdf_path = _validated_pdf_path(project_id)

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                'inline; filename="PROJECT_DOCUMENTATION.pdf"'
            )
        },
    )


@router.get(
    "/generated-docs/{project_id}/download",
    response_class=FileResponse,
)
def download_generated_docs(
    project_id: str,
) -> FileResponse:
    """Download the generated project documentation PDF."""
    pdf_path = _validated_pdf_path(project_id)

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="PROJECT_DOCUMENTATION.pdf",
    )