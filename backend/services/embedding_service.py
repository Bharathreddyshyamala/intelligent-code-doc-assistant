# backend/services/embedding_service.py
import chromadb
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, persist_directory: str = "./vector_store"):
        # Load the specific embedding model required by the architecture
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        # Initialize ChromaDB to save to your local folder
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="codebase_docs")

    def store_chunks(self, chunks: list[dict]):
        """
        Generates dense embeddings for code chunks and stores them in ChromaDB.
        """
        if not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            # The actual text to be vectorized and searched
            documents.append(chunk["code_chunk"])
            
            # Metadata for filtering later
            metadatas.append({
                "file_name": chunk["file_name"],
                "entity_name": chunk["entity_name"],
                "entity_type": chunk["entity_type"]
            })
            
            # Create a unique ID for the vector DB
            ids.append(f"{chunk['file_name']}_{chunk['entity_name']}_{idx}")

        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()

        # Add to the database
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully stored {len(chunks)} chunks in the vector database.")