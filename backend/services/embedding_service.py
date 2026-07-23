import os
from typing import List, Sequence

from dotenv import load_dotenv

from .ollama_service import (
    OLLAMA_EMBED_MODEL,
    OllamaServiceError,
    generate_embeddings,
)


load_dotenv()


# ---------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------

EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "32",
    )
)


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------


class EmbeddingServiceError(RuntimeError):
    """
    Raised when document or query embedding fails.
    """


# ---------------------------------------------------------
# Document embeddings
# ---------------------------------------------------------


def embed_documents(
    documents: Sequence[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """
    Generate one embedding vector for every document.

    Responsibilities of this function:

    1. Validate the documents.
    2. Remove unnecessary surrounding whitespace.
    3. Divide the documents into batches.
    4. Ask ollama_service to generate vectors.
    5. Combine and return all generated vectors.
    """

    if not documents:
        return []

    if batch_size <= 0:
        raise ValueError(
            "Embedding batch size must be greater than zero."
        )

    cleaned_documents = [
        document.strip()
        for document in documents
    ]

    empty_document_indexes = [
        index
        for index, document in enumerate(
            cleaned_documents
        )
        if not document
    ]

    if empty_document_indexes:
        raise ValueError(
            "Cannot generate embeddings for empty content. "
            "Empty document indexes: "
            f"{empty_document_indexes}"
        )

    all_embeddings: List[List[float]] = []

    for start_index in range(
        0,
        len(cleaned_documents),
        batch_size,
    ):
        batch = cleaned_documents[
            start_index:
            start_index + batch_size
        ]

        try:
            batch_embeddings = generate_embeddings(
                texts=batch,
                model=OLLAMA_EMBED_MODEL,
            )

        except OllamaServiceError as exc:
            raise EmbeddingServiceError(
                "Unable to generate document embeddings. "
                f"{exc}"
            ) from exc

        if len(batch_embeddings) != len(batch):
            raise EmbeddingServiceError(
                "The embedding service did not return one "
                "vector for every document in the batch."
            )

        all_embeddings.extend(
            batch_embeddings
        )

    if len(all_embeddings) != len(cleaned_documents):
        raise EmbeddingServiceError(
            "The total number of generated embeddings does "
            "not match the total number of documents."
        )

    return all_embeddings


# ---------------------------------------------------------
# Query embedding
# ---------------------------------------------------------


def embed_query(
    query: str,
) -> List[float]:
    """
    Generate one embedding vector for a RAG search query.

    This uses the same embedding model used when project
    chunks are indexed.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "The retrieval query cannot be empty."
        )

    embeddings = embed_documents(
        documents=[cleaned_query],
        batch_size=1,
    )

    if not embeddings:
        raise EmbeddingServiceError(
            "No embedding was generated for the query."
        )

    return embeddings[0]