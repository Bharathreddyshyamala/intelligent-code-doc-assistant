import json
import os
import re
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4
from zipfile import BadZipFile, ZipFile


# backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]

# backend/temp_repos/
TEMP_REPOS_DIR = BACKEND_DIR / "temp_repos"

# The current sprint uses a Python AST parser.
# Add more extensions later when parsers for other languages are implemented.
SUPPORTED_EXTENSIONS = {
    ".py",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    ".next",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    "target",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
}

VALID_SOURCE_TYPES = {
    "local",
    "upload",
    "github",
}

PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

# Optional security restriction.
# Example:
# export LOCAL_PROJECTS_ROOT=/Users/bharath/projects
LOCAL_PROJECTS_ROOT_VALUE = os.getenv("LOCAL_PROJECTS_ROOT")
LOCAL_PROJECTS_ROOT = (
    Path(LOCAL_PROJECTS_ROOT_VALUE).expanduser().resolve()
    if LOCAL_PROJECTS_ROOT_VALUE
    else None
)

TEMP_REPOS_DIR.mkdir(parents=True, exist_ok=True)


def validate_project_id(project_id: str) -> None:
    """
    Validate a project ID before using it in a filesystem path.

    Project IDs are UUID values without hyphens.
    Example:
        4fb8be61f177409d9f054ea7bdc51e48
    """

    if not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError("Invalid project_id")


def get_project_directory(project_id: str) -> Path:
    """
    Return:
        backend/temp_repos/<project_id>/
    """

    validate_project_id(project_id)
    return TEMP_REPOS_DIR / project_id


def get_source_directory(project_id: str) -> Path:
    """
    Return:
        backend/temp_repos/<project_id>/source/
    """

    return get_project_directory(project_id) / "source"


def get_metadata_path(project_id: str) -> Path:
    """
    Return:
        backend/temp_repos/<project_id>/metadata.json
    """

    return get_project_directory(project_id) / "metadata.json"


def get_ast_path(project_id: str) -> Path:
    """
    Return:
        backend/temp_repos/<project_id>/ast.json
    """

    return get_project_directory(project_id) / "ast.json"


def get_chunks_path(project_id: str) -> Path:
    """
    Return:
        backend/temp_repos/<project_id>/chunks.json

    Member 2 will use this path later.
    """

    return get_project_directory(project_id) / "chunks.json"


def create_project_workspace() -> Tuple[str, Path]:
    """
    Create a new project folder.

    Returns:
        project_id
        source_directory

    Example:
        (
            "4fb8be61f177409d9f054ea7bdc51e48",
            Path("backend/temp_repos/.../source")
        )
    """

    project_id = uuid4().hex
    source_directory = get_source_directory(project_id)

    source_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return project_id, source_directory


def delete_project_workspace(project_id: str) -> None:
    """
    Delete the complete project folder when ingestion fails.
    """

    project_directory = get_project_directory(project_id)

    shutil.rmtree(
        project_directory,
        ignore_errors=True,
    )


def resolve_local_project_path(path_value: str) -> Path:
    """
    Resolve and validate a local project path.

    When LOCAL_PROJECTS_ROOT is configured, users may only access
    folders inside that directory.
    """

    source_path = Path(path_value).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            f"Local project path does not exist: {source_path}"
        )

    if not source_path.is_dir():
        raise ValueError(
            "The local project path must point to a directory"
        )

    if LOCAL_PROJECTS_ROOT is not None:
        is_allowed = (
            source_path == LOCAL_PROJECTS_ROOT
            or LOCAL_PROJECTS_ROOT in source_path.parents
        )

        if not is_allowed:
            raise PermissionError(
                "The local project must be inside "
                f"{LOCAL_PROJECTS_ROOT}"
            )

    return source_path


def should_ignore(relative_path: Path) -> bool:
    """
    Return True when a file belongs to an ignored directory.
    """

    return any(
        path_part in IGNORED_DIRECTORIES
        for path_part in relative_path.parts
    )


def is_supported_code_file(file_path: Path) -> bool:
    """
    Check whether the file is supported by the current parser.
    """

    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def copy_supported_code_files(
    source_directory: Path,
    destination_directory: Path,
) -> int:
    """
    Copy supported source-code files into the project workspace.

    The original folder structure is preserved.

    Example:

        source project:
            calculator/main.py
            calculator/services/math_service.py

        destination:
            source/main.py
            source/services/math_service.py

    Returns:
        Number of copied files.
    """

    if not source_directory.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_directory}"
        )

    if not source_directory.is_dir():
        raise ValueError(
            "The source path must point to a directory"
        )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_count = 0

    for source_file in source_directory.rglob("*"):
        if not source_file.is_file():
            continue

        # Symbolic links are skipped to avoid reading files
        # outside the selected project.
        if source_file.is_symlink():
            continue

        relative_path = source_file.relative_to(
            source_directory
        )

        if should_ignore(relative_path):
            continue

        if not is_supported_code_file(source_file):
            continue

        destination_file = (
            destination_directory / relative_path
        )

        destination_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_file,
            destination_file,
        )

        file_count += 1

    return file_count


def safe_extract_zip(
    zip_path: Path,
    destination_directory: Path,
    maximum_files: int = 10_000,
    maximum_extracted_size: int = 500 * 1024 * 1024,
) -> None:
    """
    Safely extract a ZIP file.

    Security checks:
    - Reject absolute paths
    - Reject ../ directory traversal
    - Reject symbolic links
    - Limit the number of extracted files
    - Limit total extracted size
    """

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_destination = destination_directory.resolve()

    try:
        with ZipFile(zip_path, "r") as archive:
            members = archive.infolist()

            if len(members) > maximum_files:
                raise ValueError(
                    "The ZIP file contains too many files"
                )

            total_size = sum(
                member.file_size
                for member in members
            )

            if total_size > maximum_extracted_size:
                raise ValueError(
                    "The extracted ZIP content is too large"
                )

            for member in members:
                normalized_name = member.filename.replace(
                    "\\",
                    "/",
                )

                member_path = PurePosixPath(normalized_name)

                if member_path.is_absolute():
                    raise ValueError(
                        "The ZIP file contains an unsafe "
                        f"absolute path: {member.filename}"
                    )

                if ".." in member_path.parts:
                    raise ValueError(
                        "The ZIP file contains an unsafe path: "
                        f"{member.filename}"
                    )

                unix_mode = member.external_attr >> 16

                if stat.S_ISLNK(unix_mode):
                    raise ValueError(
                        "Symbolic links are not allowed "
                        "inside uploaded ZIP files"
                    )

                target_path = (
                    destination_directory
                    / Path(*member_path.parts)
                ).resolve()

                is_safe_destination = (
                    target_path == resolved_destination
                    or resolved_destination in target_path.parents
                )

                if not is_safe_destination:
                    raise ValueError(
                        "The ZIP file contains an unsafe "
                        f"destination: {member.filename}"
                    )

                if member.is_dir():
                    target_path.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    continue

                target_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with archive.open(member, "r") as source:
                    with target_path.open("wb") as destination:
                        shutil.copyfileobj(
                            source,
                            destination,
                        )

    except BadZipFile as exc:
        raise ValueError(
            "The uploaded file is not a valid ZIP file"
        ) from exc


def count_code_files(source_directory: Path) -> int:
    """
    Count supported source-code files.
    """

    if not source_directory.exists():
        return 0

    file_count = 0

    for file_path in source_directory.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(
            source_directory
        )

        if should_ignore(relative_path):
            continue

        if is_supported_code_file(file_path):
            file_count += 1

    return file_count


def save_metadata(
    project_id: str,
    source_type: str,
    source_value: str,
    file_count: int,
    status: str,
) -> Dict[str, Any]:
    """
    Create metadata.json for an ingested project.

    Member 2 depends on:
        project_id
        source_type
    """

    validate_project_id(project_id)

    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source_type: {source_type}"
        )

    current_time = datetime.now(
        timezone.utc
    ).isoformat()

    metadata: Dict[str, Any] = {
        "project_id": project_id,
        "source_type": source_type,
        "source_value": source_value,
        "file_count": file_count,
        "parsed_file_count": 0,
        "parsing_error_count": 0,
        "chunk_count": 0,
        "status": status,
        "created_at": current_time,
        "updated_at": current_time,
    }

    metadata_path = get_metadata_path(project_id)

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


def read_metadata(project_id: str) -> Dict[str, Any]:
    """
    Read metadata for an existing project.
    """

    metadata_path = get_metadata_path(project_id)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Project '{project_id}' was not found"
        )

    try:
        return json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Project metadata is invalid for '{project_id}'"
        ) from exc


def update_metadata(
    project_id: str,
    **updates: Any,
) -> Dict[str, Any]:
    """
    Update metadata after parsing, chunking, or indexing.
    """

    metadata = read_metadata(project_id)

    protected_fields = {
        "project_id",
        "created_at",
    }

    for key, value in updates.items():
        if key not in protected_fields:
            metadata[key] = value

    metadata["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    get_metadata_path(project_id).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata

def find_existing_project(
    source_type: str,
    source_value: str,
) -> Optional[dict]:
    """
    Find an existing project using its source type and source value.
    """

    TEMP_REPOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if source_type == "local":
        normalized_source = str(
            Path(source_value).expanduser().resolve()
        )
    else:
        normalized_source = source_value.strip()

    for project_directory in TEMP_REPOS_DIR.iterdir():
        if not project_directory.is_dir():
            continue

        metadata_path = project_directory / "metadata.json"

        if not metadata_path.exists():
            continue

        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except (json.JSONDecodeError, OSError):
            continue

        saved_source = metadata.get("source_value")

        if source_type == "local" and saved_source:
            saved_source = str(
                Path(saved_source)
                .expanduser()
                .resolve()
            )

        if (
            metadata.get("source_type") == source_type
            and saved_source == normalized_source
        ):
            return metadata

    return None