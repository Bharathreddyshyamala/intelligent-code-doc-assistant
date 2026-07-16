from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, HttpUrl

from services.file_scanner import (
    copy_supported_code_files,
    create_project_workspace,
    delete_project_workspace,
    find_existing_project,
    resolve_local_project_path,
    safe_extract_zip,
    save_metadata,
)
from services.github_ingestion import (
    ingest_github_repository,
)


router = APIRouter(
    tags=["Ingestion"],
)

MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)


class LocalIngestionRequest(BaseModel):
    path: str = Field(
        ...,
        min_length=1,
        description="Path to a local project folder",
    )


class GitHubIngestionRequest(BaseModel):
    repo_url: HttpUrl
    branch: Optional[str] = Field(
        default=None,
        description="Optional GitHub branch",
    )


class ProjectResponse(BaseModel):
    project_id: str
    file_count: int
    status: str


@router.post(
    "/ingest-local",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_local(
    request: LocalIngestionRequest,
) -> ProjectResponse:
    """
    Ingest a local project folder.

    The folder must exist on the machine where FastAPI runs.
    """

    project_id: Optional[str] = None

    try:
        local_project_path = resolve_local_project_path(
            request.path
        )
        existing_project = find_existing_project(
            source_type="local",
            source_value=str(local_project_path),
        )

        if existing_project:
            return ProjectResponse(
                project_id=existing_project["project_id"],
                file_count=existing_project["file_count"],
                status="already_ingested",
            )

        project_id, source_directory = (
            create_project_workspace()
        )

        file_count = copy_supported_code_files(
            source_directory=local_project_path,
            destination_directory=source_directory,
        )

        if file_count == 0:
            raise ValueError(
                "No supported Python source files were found"
            )

        save_metadata(
            project_id=project_id,
            source_type="local",
            source_value=str(local_project_path),
            file_count=file_count,
            status="ingested",
        )

        return ProjectResponse(
            project_id=project_id,
            file_count=file_count,
            status="ingested",
        )

    except FileNotFoundError as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except PermissionError as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local project ingestion failed: {exc}",
        ) from exc


@router.post(
    "/ingest-upload",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_upload(
    file: UploadFile = File(...),
) -> ProjectResponse:
    """
    Ingest a project uploaded as a ZIP file.
    """

    project_id: Optional[str] = None
    temporary_zip_path: Optional[Path] = None

    original_filename = (
        file.filename or "uploaded-project.zip"
    )

    if Path(original_filename).suffix.lower() != ".zip":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only ZIP files are supported",
        )

    uploaded_size = 0

    try:
        project_id, source_directory = (
            create_project_workspace()
        )

        # Stream the uploaded ZIP into a temporary file.
        with NamedTemporaryFile(
            mode="wb",
            suffix=".zip",
            delete=False,
        ) as temporary_zip:
            temporary_zip_path = Path(
                temporary_zip.name
            )

            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                uploaded_size += len(chunk)

                if uploaded_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "The uploaded ZIP exceeds the "
                            f"{MAX_UPLOAD_SIZE_MB} MB limit"
                        ),
                    )

                temporary_zip.write(chunk)

        # Extract into a temporary folder first.
        # Only supported code files are copied to source/.
        with TemporaryDirectory(
            prefix="code-doc-upload-"
        ) as extracted_directory_value:
            extracted_directory = Path(
                extracted_directory_value
            )

            safe_extract_zip(
                zip_path=temporary_zip_path,
                destination_directory=extracted_directory,
            )

            file_count = copy_supported_code_files(
                source_directory=extracted_directory,
                destination_directory=source_directory,
            )

        if file_count == 0:
            raise ValueError(
                "The uploaded ZIP contains no supported "
                "Python source files"
            )

        save_metadata(
            project_id=project_id,
            source_type="upload",
            source_value=original_filename,
            file_count=file_count,
            status="ingested",
        )

        return ProjectResponse(
            project_id=project_id,
            file_count=file_count,
            status="ingested",
        )

    except HTTPException:
        if project_id:
            delete_project_workspace(project_id)

        raise

    except ValueError as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Uploaded project ingestion failed: {exc}",
        ) from exc

    finally:
        await file.close()

        if temporary_zip_path is not None:
            temporary_zip_path.unlink(
                missing_ok=True
            )


@router.post(
    "/ingest-github",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_github(
    request: GitHubIngestionRequest,
) -> ProjectResponse:
    """
    Ingest a public GitHub repository.
    """

    project_id: Optional[str] = None

    try:
        project_id, source_directory = (
            create_project_workspace()
        )

        repository_url = str(request.repo_url)

        file_count = ingest_github_repository(
            repo_url=repository_url,
            destination_directory=source_directory,
            branch=request.branch,
        )

        if file_count == 0:
            raise ValueError(
                "The GitHub repository contains no supported "
                "Python source files"
            )

        save_metadata(
            project_id=project_id,
            source_type="github",
            source_value=repository_url,
            file_count=file_count,
            status="ingested",
        )

        return ProjectResponse(
            project_id=project_id,
            file_count=file_count,
            status="ingested",
        )

    except ValueError as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        if project_id:
            delete_project_workspace(project_id)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub project ingestion failed: {exc}",
        ) from exc