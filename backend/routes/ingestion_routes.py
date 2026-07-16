# backend/routes/ingestion_routes.py

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict
import os

from services.file_scanner import FileScanner
from services.github_ingestion import GitHubIngestionService

router = APIRouter(
    prefix="/api/ingest",
    tags=["ingestion"]
)

# Initialize services
file_scanner = FileScanner()
github_ingestion_service = GitHubIngestionService(file_scanner=file_scanner)

# Pydantic request models
class LocalScanRequest(BaseModel):
    directory_path: str = Field(..., description="Absolute or relative path to the local project directory to scan")

class GitHubIngestRequest(BaseModel):
    github_url: str = Field(..., description="The HTTPS or SSH GitHub repository URL")

# Pydantic response models
class FileScanItem(BaseModel):
    file_path: str
    file_name: str
    extension: str
    raw_code: str

class IngestionResponse(BaseModel):
    status: str
    message: str
    count: int
    files: List[FileScanItem]

@router.post("/local", response_model=IngestionResponse, status_code=status.HTTP_200_OK)
def scan_local_directory(request: LocalScanRequest):
    """
    Scans a local project folder and returns detected source code files.
    """
    path = request.directory_path
    
    # Simple check for safety: resolve absolute path and make sure it is valid
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The specified directory path does not exist: {path}"
        )
    if not os.path.isdir(path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The specified path is not a directory: {path}"
        )
        
    try:
        scanned_files = file_scanner.scan_directory(path)
        # Convert absolute paths to relative paths relative to the directory path scanned, for cleaner output
        for file_info in scanned_files:
            rel_path = os.path.relpath(file_info["file_path"], path)
            file_info["file_path"] = rel_path.replace(os.path.sep, "/")

        return IngestionResponse(
            status="success",
            message=f"Successfully scanned directory: {path}",
            count=len(scanned_files),
            files=scanned_files
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while scanning: {str(e)}"
        )

@router.post("/github", response_model=IngestionResponse, status_code=status.HTTP_200_OK)
def ingest_github_repository(request: GitHubIngestRequest):
    """
    Clones a GitHub repository into a temporary folder, scans it, and returns the source code files.
    """
    url = request.github_url
    
    if not github_ingestion_service.validate_github_url(url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid GitHub URL: {url}. Must be a valid HTTPS or SSH GitHub repository link."
        )
        
    try:
        scanned_files = github_ingestion_service.scan_repo(url)
        return IngestionResponse(
            status="success",
            message=f"Successfully scanned GitHub repository: {url}",
            count=len(scanned_files),
            files=scanned_files
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process GitHub repository. Error: {str(e)}"
        )
