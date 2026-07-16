from services.file_scanner import FileScanner
from services.github_ingestion import GitHubIngestionService

def check_local():
    print("--- Testing Local Scanner ---")
    scanner = FileScanner()
    # Pointing it at your backend folder to see what it finds
    results = scanner.scan_directory('./backend')
    print(f"Success! Found {len(results)} valid source files.")
    if results:
        print(f"Sample file: {results[0]['file_name']} | Size: {len(results[0]['raw_code'])} chars\n")

def check_github():
    print("--- Testing GitHub Scanner ---")
    scanner = GitHubIngestionService()
    # A safe, public repository to test the clone logic
    test_url = "https://github.com/pallets/flask" 
    
    try:
        results = scanner.scan_repo(test_url)
        print(f"Success! Cloned and scanned {len(results)} files.")
        if results:
            print(f"Sample file: {results[0]['file_name']} | Size: {len(results[0]['raw_code'])} chars")
    except Exception as e:
        print(f"GitHub Error: {e}")

if __name__ == "__main__":
    check_local()
    check_github()