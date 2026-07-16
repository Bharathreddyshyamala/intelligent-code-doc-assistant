from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from urllib.parse import urlparse

from git import GitCommandError, Repo

from services.file_scanner import (
    copy_supported_code_files,
)


def validate_github_url(repo_url: str) -> None:
    """
    Validate a public HTTPS GitHub repository URL.
    """

    parsed_url = urlparse(repo_url)

    if parsed_url.scheme != "https":
        raise ValueError(
            "Only HTTPS GitHub repository URLs are supported"
        )

    if parsed_url.hostname not in {
        "github.com",
        "www.github.com",
    }:
        raise ValueError(
            "The repository URL must point to github.com"
        )

    if parsed_url.username or parsed_url.password:
        raise ValueError(
            "Credentials must not be included in the repository URL"
        )

    path_parts = [
        part
        for part in parsed_url.path.split("/")
        if part
    ]

    if len(path_parts) < 2:
        raise ValueError(
            "Invalid GitHub repository URL. "
            "Expected a URL containing an owner and repository."
        )


def ingest_github_repository(
    repo_url: str,
    destination_directory: Path,
    branch: Optional[str] = None,
) -> int:
    """
    Clone a GitHub repository and copy supported source files
    into the project's source directory.

    Returns:
        Number of copied code files.
    """

    validate_github_url(repo_url)

    with TemporaryDirectory(
        prefix="code-doc-github-"
    ) as temporary_directory:
        clone_directory = (
            Path(temporary_directory) / "repository"
        )

        clone_arguments = {
            "url": repo_url,
            "to_path": str(clone_directory),
            "depth": 1,
        }

        if branch:
            clone_arguments["branch"] = branch

        try:
            repository = Repo.clone_from(
                **clone_arguments
            )

            repository.close()

        except GitCommandError as exc:
            raise ValueError(
                "Unable to clone the GitHub repository. "
                "Check the repository URL, branch, and repository access."
            ) from exc

        return copy_supported_code_files(
            source_directory=clone_directory,
            destination_directory=destination_directory,
        )