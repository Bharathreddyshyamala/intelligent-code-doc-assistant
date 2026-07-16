# backend/services/file_scanner.py

import os
from typing import List, Dict

class FileScanner:
    """
    Service to scan a directory path recursively, filtering for source code files
    and ignoring standard virtual environment and VCS directories.
    """
    def __init__(self, allowed_extensions: List[str] = None, ignored_dirs: List[str] = None):
        # Default target extensions for code files
        if allowed_extensions is None:
            self.allowed_extensions = {'.py', '.js', '.java', '.cpp', '.ts'}
        else:
            self.allowed_extensions = {ext.lower() for ext in allowed_extensions}
            
        # Default ignored directories
        if ignored_dirs is None:
            self.ignored_dirs = {'.git', 'node_modules', 'venv', '.venv', '__pycache__'}
        else:
            self.ignored_dirs = set(ignored_dirs)

    def scan_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """
        Scans a given directory recursively.
        
        Args:
            directory_path: The filesystem path to scan.
            
        Returns:
            A list of dicts, each with keys: 'file_path', 'file_name', 'extension', 'raw_code'.
        """
        results = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Path does not exist: {directory_path}")
            
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")

        # os.walk allows modifying dirs in-place to prune search paths
        for root, dirs, files in os.walk(directory_path, topdown=True):
            # Modify dirs in-place to exclude ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            for file in files:
                _, ext = os.path.splitext(file)
                ext_lower = ext.lower()
                if ext_lower in self.allowed_extensions:
                    full_path = os.path.abspath(os.path.join(root, file))
                    try:
                        # Using 'utf-8' with errors='replace' to avoid crashes on non-utf-8 files
                        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                            raw_code = f.read()
                    except Exception as e:
                        # If a file is completely unreadable, skip it
                        continue
                        
                    results.append({
                        "file_path": full_path,
                        "file_name": file,
                        "extension": ext_lower,
                        "raw_code": raw_code
                    })
                    
        return results
