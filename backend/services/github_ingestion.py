# backend/services/github_ingestion.py

import os
import re
import shutil
import stat
import subprocess
import tempfile
from typing import List, Dict
from services.file_scanner import FileScanner

class GitHubIngestionService:
    """
    Service to validate GitHub repository URLs, clone them into a temporary folder,
    and scan their source code files using FileScanner.
    """
    def __init__(self, file_scanner: FileScanner = None):
        self.file_scanner = file_scanner or FileScanner()
        # Regex to validate HTTP, HTTPS, and SSH GitHub repository URLs
        self.github_url_pattern = re.compile(
            r'^(https?://github\.com/|git@github\.com:)[a-zA-Z0-9_-]+/[a-zA-Z0-9_\.-]+?(\.git)?/?$'
        )

    def validate_github_url(self, url: str) -> bool:
        """
        Validates whether the provided string is a valid GitHub repository URL.
        """
        if not url:
            return False
        return bool(self.github_url_pattern.match(url.strip()))

    def _clone_repo(self, url: str, clone_path: str) -> None:
        """
        Clones the github repo into the specified path.
        """
        try:
            # Run git clone command
            result = subprocess.run(
                ["git", "clone", url.strip(), clone_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git clone failed. Error: {e.stderr or e.stdout}")

    def _cleanup_dir(self, path: str) -> None:
        """
        Safely removes a directory, resolving Windows read-only file permission issues (common with .git files).
        """
        def remove_readonly(func, file_path, excinfo):
            # Change file permission to writeable and retry the delete operation
            try:
                os.chmod(file_path, stat.S_IWRITE)
                func(file_path)
            except Exception:
                # If it still fails, let the error propagate or ignore if it's fine
                pass

        if os.path.exists(path):
            shutil.rmtree(path, onerror=remove_readonly)

    def scan_repo(self, url: str) -> List[Dict[str, str]]:
        """
        Clones a GitHub repository to a temporary directory, scans the source code files,
        cleans up the temporary directory, and returns the files.
        """
        if not self.validate_github_url(url):
            raise ValueError(f"Invalid GitHub URL: {url}")

        # Create a unique temporary directory
        temp_dir = tempfile.mkdtemp(prefix="github_ingest_")
        
        try:
            # Clone the repository
            self._clone_repo(url, temp_dir)
            # Scan files using the file scanner
            files = self.file_scanner.scan_directory(temp_dir)
            
            # Make the file paths relative to the temp_dir so they don't contain the absolute temp folder prefix
            for file_info in files:
                rel_path = os.path.relpath(file_info["file_path"], temp_dir)
                # Normalize path separators to forward slash for consistency
                file_info["file_path"] = rel_path.replace(os.path.sep, "/")
                
            return files
        finally:
            # Ensure cleanup happens even if scan or clone fails
            self._cleanup_dir(temp_dir)
