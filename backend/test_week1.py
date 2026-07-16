# backend/test_week1.py

import os
import tempfile
import shutil
from services.file_scanner import FileScanner
from services.github_ingestion import GitHubIngestionService

def run_file_scanner_test():
    print("--- 1. Testing File Scanner ---")
    scanner = FileScanner()
    
    # Create a temporary local directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temp directory for scan: {temp_dir}")
        
        # Create valid files
        valid_files = {
            "app.py": "print('hello world')",
            "utils.js": "const add = (a, b) => a + b;",
            "Main.java": "public class Main { }",
            "helper.ts": "export const value = 42;",
            "core.cpp": "#include <iostream>\nint main() { return 0; }"
        }
        for name, content in valid_files.items():
            with open(os.path.join(temp_dir, name), 'w', encoding='utf-8') as f:
                f.write(content)

        # Create ignored directories and files inside them
        ignored_dirs = [".git", "node_modules", "venv", ".venv", "__pycache__"]
        for d in ignored_dirs:
            ignored_dir_path = os.path.join(temp_dir, d)
            os.makedirs(ignored_dir_path, exist_ok=True)
            with open(os.path.join(ignored_dir_path, "secret.py"), 'w', encoding='utf-8') as f:
                f.write("print('should be ignored')")

        # Create invalid file extensions at root
        with open(os.path.join(temp_dir, "notes.txt"), 'w', encoding='utf-8') as f:
            f.write("Just some notes.")
        with open(os.path.join(temp_dir, "doc.md"), 'w', encoding='utf-8') as f:
            f.write("# Documentation")

        # Run scanner
        scanned_files = scanner.scan_directory(temp_dir)
        
        print(f"Scanned files count: {len(scanned_files)} (Expected: 5)")
        
        # Verify
        detected_names = {f["file_name"] for f in scanned_files}
        expected_names = set(valid_files.keys())
        assert detected_names == expected_names, f"Mismatch: {detected_names} vs {expected_names}"
        
        print("Scanned files details:")
        for f in scanned_files:
            print(f" - Name: {f['file_name']}, Ext: {f['extension']}, Raw length: {len(f['raw_code'])} chars")
            
    print("File Scanner Test: SUCCESS\n")


def run_github_ingestion_test():
    print("--- 2. Testing GitHub Ingestion ---")
    service = GitHubIngestionService()
    
    # Test URL validation
    valid_urls = [
        "https://github.com/owner/repo",
        "https://github.com/owner/repo.git",
        "git@github.com:owner/repo.git",
        "http://github.com/owner/repo"
    ]
    invalid_urls = [
        "https://github.com/owner",
        "https://google.com/owner/repo",
        "invalid-url"
    ]
    
    print("Checking URL validation:")
    for url in valid_urls:
        assert service.validate_github_url(url) is True, f"Failed on valid URL: {url}"
    for url in invalid_urls:
        assert service.validate_github_url(url) is False, f"Failed on invalid URL: {url}"
    print(" - URL validation: SUCCESS")

    # Mock the cloning to perform a fast local check
    dummy_files = {
        "index.js": "console.log('Ingested GitHub code')",
        "process.py": "import sys"
    }

    def mock_clone(url, target_path):
        os.makedirs(target_path, exist_ok=True)
        for name, content in dummy_files.items():
            with open(os.path.join(target_path, name), 'w', encoding='utf-8') as f:
                f.write(content)

    # Backup the real clone method and replace it
    real_clone = service._clone_repo
    service._clone_repo = mock_clone

    # Run scan
    results = service.scan_repo("https://github.com/owner/repo")
    
    print(f"Scanned repo files count: {len(results)} (Expected: 2)")
    
    # Check paths and clean-up
    for r in results:
        print(f" - Path: {r['file_path']}, Name: {r['file_name']}, Ext: {r['extension']}")
        # Verify relative path formatting
        assert not os.path.isabs(r['file_path']), f"Path should be relative: {r['file_path']}"
        
    # Restore clone method
    service._clone_repo = real_clone
    
    print("GitHub Ingestion Test: SUCCESS\n")

if __name__ == "__main__":
    print("=== Starting Ingestion Services Self-Test ===")
    run_file_scanner_test()
    run_github_ingestion_test()
    print("=== All Ingestion Service Self-Tests Passed! ===")
