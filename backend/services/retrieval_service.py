from typing import Any, Dict, List

from .embedding_service import embed_query
from .file_scanner import validate_project_id
from .indexer import get_chroma_collection


class RetrievalServiceError(RuntimeError):
    """
    Raised when code retrieval from ChromaDB fails.
    """


def get_first_result_batch(
    results: Dict[str, Any],
    field_name: str,
) -> List[Any]:
    """
    ChromaDB query results are returned as batches.

    Since this application sends one query at a time,
    this helper returns the first batch.
    """

    values = results.get(field_name)

    if not values:
        return []

    first_batch = values[0]

    if first_batch is None:
        return []

    return list(first_batch)


def retrieve_code_chunks(
    project_id: str,
    query: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Retrieve the code chunks most relevant to a question.

    Steps:
    1. Validate the project.
    2. Embed the user question.
    3. Search ChromaDB.
    4. Restrict results to the current project.
    5. Return documents, metadata, and distances.
    """

    validate_project_id(
        project_id
    )

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "The retrieval query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    # Generate a vector for the user's question.
    query_embedding = embed_query(
        cleaned_query
    )

    collection = get_chroma_collection()

    try:
        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            where={
                "project_id": project_id,
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    except Exception as exc:
        raise RetrievalServiceError(
            "Unable to retrieve code chunks from "
            f"ChromaDB: {exc}"
        ) from exc

    ids = get_first_result_batch(
        results,
        "ids",
    )

    documents = get_first_result_batch(
        results,
        "documents",
    )

    metadatas = get_first_result_batch(
        results,
        "metadatas",
    )

    distances = get_first_result_batch(
        results,
        "distances",
    )

    retrieved_chunks: List[Dict[str, Any]] = []

    for index, chunk_id in enumerate(ids):
        document = (
            documents[index]
            if index < len(documents)
            else ""
        )

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else 0.0
        )

        retrieved_chunks.append(
            {
                "rank": index + 1,
                "chunk_id": chunk_id,
                "document": document or "",
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )

    return retrieved_chunks