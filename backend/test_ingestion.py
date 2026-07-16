# backend/test_ingestion.py

import os
from services.chunker import CodeChunker
from services.embedding_service import EmbeddingService

def run_test():
    print("--- Starting Ingestion Pipeline Test ---\n")
    
    # 1. Initialize your services
    print("Initializing services...")
    chunker = CodeChunker()
    # This will create/connect to the ChromaDB database in backend/vector_store/
    embedder = EmbeddingService(persist_directory="./vector_store")

    # 2. Define some mock code to simulate the output of your file scanner
    sample_code = """
class DataPipeline:
    def __init__(self, data_source):
        self.data_source = data_source

    def clean_data(self, raw_data):
        # Removes null values and formats strings
        cleaned = [d.strip() for d in raw_data if d is not None]
        return cleaned

    def execute_pipeline(self):
        data = [" apple ", None, "banana"]
        return self.clean_data(data)
"""
    
    # 3. Test the AST Chunker
    print("\nChunking mock code...")
    chunks = chunker.chunk_file(raw_code=sample_code, file_name="mock_pipeline.py")
    
    print(f"Successfully extracted {len(chunks)} chunks:")
    for chunk in chunks:
        print(f" -> {chunk['entity_type']}: {chunk['entity_name']} (Lines {chunk['start_line']}-{chunk['end_line']})")

    # 4. Test the Embedding Service
    print("\nGenerating embeddings and storing in ChromaDB...")
    embedder.store_chunks(chunks)

    # 5. Verify the Retrieval
    print("\nVerifying database storage by running a test query...")
    # We ask a natural language question to see if the semantic search works
    test_query = "How do I format the strings and remove nulls?"
    
    query_embeddings = embedder.model.encode([test_query]).tolist()
    results = embedder.collection.query(
        query_embeddings=query_embeddings,
        n_results=1
    )

    print(f"\nQuery: '{test_query}'")
    if results['documents'] and results['documents'][0]:
        print("Top Result Found:")
        print("-" * 40)
        print(results['documents'][0][0])  # Prints the raw code chunk that matched
        print("-" * 40)
        print(f"Metadata: {results['metadatas'][0][0]}")
    else:
        print("No results found. Something went wrong with the embedding storage.")

if __name__ == "__main__":
    run_test()