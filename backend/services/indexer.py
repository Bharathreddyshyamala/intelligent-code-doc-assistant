import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv

from .chunker import (
    chunk_project,
    load_ast,
)
from .embedding_service import (
    EmbeddingServiceError,
    embed_documents,
)
from .file_scanner import (
    update_metadata,
    validate_project_id,
)
from .ollama_service import (
    OLLAMA_EMBED_MODEL,
)


load_dotenv()


# ---------------------------------------------------------
# ChromaDB configuration
# ---------------------------------------------------------

BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]


def resolve_chroma_path() -> Path:
    """
    Resolve CHROMA_DB_PATH relative to the backend directory
    when a relative path is configured.
    """

    configured_path = Path(
        os.getenv(
            "CHROMA_DB_PATH",
            "vector_store/chroma_db",
        )
    ).expanduser()

    if configured_path.is_absolute():
        return configured_path

    return (
        BACKEND_DIRECTORY / configured_path
    ).resolve()


CHROMA_DB_PATH = resolve_chroma_path()

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "code_chunks",
)

CHROMA_BATCH_SIZE = int(
    os.getenv(
        "CHROMA_BATCH_SIZE",
        "100",
    )
)


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------


class IndexingError(RuntimeError):
    """
    Base exception for indexing failures.
    """


class VectorStoreError(IndexingError):
    """
    Raised when ChromaDB cannot store or manage vectors.
    """


# ---------------------------------------------------------
# ChromaDB connection
# ---------------------------------------------------------


def get_chroma_collection() -> Any:
    """
    Create or load the persistent ChromaDB collection.
    """

    try:
        CHROMA_DB_PATH.mkdir(
            parents=True,
            exist_ok=True,
        )

        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
        )

        return client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
        )

    except Exception as exc:
        raise VectorStoreError(
            f"Unable to initialize ChromaDB: {exc}"
        ) from exc


# ---------------------------------------------------------
# ChromaDB metadata
# ---------------------------------------------------------


def sanitize_metadata(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert chunk metadata into values that ChromaDB can store.

    Missing values such as None are converted into empty
    strings. Complex values are converted into JSON strings.
    """

    sanitized: Dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""

        elif isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            sanitized[key] = value

        else:
            sanitized[key] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )

    sanitized["embedding_model"] = (
        OLLAMA_EMBED_MODEL
    )

    return sanitized


# ---------------------------------------------------------
# Embedding document preparation
# ---------------------------------------------------------


def build_embedding_document(
    chunk: Dict[str, Any],
) -> str:
    """
    Convert a code chunk into readable text for embedding.

    Metadata such as file name, symbol name, source type,
    line range, and docstring is combined with the source
    code to improve semantic retrieval.
    """

    metadata = chunk["metadata"]

    sections = [
        f"File: {metadata.get('file_path', '')}",
        f"Symbol: {metadata.get('name', '')}",
        f"Type: {metadata.get('source_type', '')}",
        (
            "Lines: "
            f"{metadata.get('start_line', '')}-"
            f"{metadata.get('end_line', '')}"
        ),
    ]

    docstring = metadata.get(
        "docstring"
    )

    if docstring:
        sections.append(
            f"Description: {docstring}"
        )

    sections.append(
        f"Source code:\n{chunk['content']}"
    )

    return "\n\n".join(
        sections
    )


# ---------------------------------------------------------
# Existing project records
# ---------------------------------------------------------


def delete_existing_project_index(
    collection: Any,
    project_id: str,
) -> None:
    """
    Remove previously indexed chunks for the project.

    The current chunker creates UUID-based chunk IDs.
    Removing previous project records prevents duplicates
    when the project is indexed again.
    """

    try:
        collection.delete(
            where={
                "project_id": project_id,
            }
        )

    except Exception as exc:
        raise VectorStoreError(
            "Unable to remove the previous project index: "
            f"{exc}"
        ) from exc


# ---------------------------------------------------------
# Store vectors in ChromaDB
# ---------------------------------------------------------


def store_chunks(
    collection: Any,
    chunks: List[Dict[str, Any]],
    documents: List[str],
    embeddings: List[List[float]],
    batch_size: int = CHROMA_BATCH_SIZE,
) -> int:
    """
    Store chunks, embedding documents, metadata, and vectors
    in ChromaDB.
    """

    if not (
        len(chunks)
        == len(documents)
        == len(embeddings)
    ):
        raise ValueError(
            "Chunk, document, and embedding counts "
            "must be equal."
        )

    if batch_size <= 0:
        raise ValueError(
            "ChromaDB batch size must be greater than zero."
        )

    indexed_count = 0

    for start_index in range(
        0,
        len(chunks),
        batch_size,
    ):
        chunk_batch = chunks[
            start_index:
            start_index + batch_size
        ]

        document_batch = documents[
            start_index:
            start_index + batch_size
        ]

        embedding_batch = embeddings[
            start_index:
            start_index + batch_size
        ]

        try:
            collection.upsert(
                ids=[
                    chunk["chunk_id"]
                    for chunk in chunk_batch
                ],
                documents=document_batch,
                metadatas=[
                    sanitize_metadata(
                        chunk["metadata"]
                    )
                    for chunk in chunk_batch
                ],
                embeddings=embedding_batch,
            )

        except Exception as exc:
            raise VectorStoreError(
                "Unable to store chunks in ChromaDB: "
                f"{exc}"
            ) from exc

        indexed_count += len(
            chunk_batch
        )

    return indexed_count


# ---------------------------------------------------------
# Complete indexing workflow
# ---------------------------------------------------------


def index_project(
    project_id: str,
) -> Dict[str, Any]:
    """
    Run the complete project indexing workflow.

    Steps:

    1. Validate the project ID.
    2. Generate chunks.json.
    3. Build readable embedding documents.
    4. Generate embeddings through embedding_service.
    5. Store vectors in ChromaDB.
    6. Update project metadata.
    """

    validate_project_id(
        project_id
    )

    try:
        update_metadata(
            project_id,
            status="chunking",
            indexing_error="",
        )

        # Step 1: Create chunks.json
        chunks = chunk_project(
            project_id
        )

        if not chunks:
            raise ValueError(
                "No chunks were created. Make sure the "
                "project was parsed and contains Python "
                "functions or classes."
            )

        ast_data = load_ast(
            project_id
        )

        parsed_files = [
            file_entry
            for file_entry in ast_data.get(
                "files",
                [],
            )
            if file_entry.get("status") == "parsed"
        ]

        # Step 2: Prepare searchable embedding documents
        documents = [
            build_embedding_document(
                chunk
            )
            for chunk in chunks
        ]

        update_metadata(
            project_id,
            status="embedding",
            chunk_count=len(chunks),
            embedding_model=OLLAMA_EMBED_MODEL,
        )

        # Step 3: Generate vectors
        embeddings = embed_documents(
            documents
        )

        if len(embeddings) != len(chunks):
            raise EmbeddingServiceError(
                "The embedding service did not return one "
                "embedding for every code chunk."
            )

        update_metadata(
            project_id,
            status="indexing",
            chunk_count=len(chunks),
            embedding_count=len(embeddings),
            embedding_model=OLLAMA_EMBED_MODEL,
        )

        # Step 4: Open ChromaDB
        collection = get_chroma_collection()

        # Step 5: Prevent duplicate records
        delete_existing_project_index(
            collection=collection,
            project_id=project_id,
        )

        # Step 6: Store vectors
        indexed_count = store_chunks(
            collection=collection,
            chunks=chunks,
            documents=documents,
            embeddings=embeddings,
        )

        update_metadata(
            project_id,
            status="indexed",
            chunk_count=len(chunks),
            embedding_count=len(embeddings),
            indexed_count=indexed_count,
            embedding_model=OLLAMA_EMBED_MODEL,
            chroma_collection=CHROMA_COLLECTION_NAME,
            indexing_error="",
        )

        return {
            "project_id": project_id,
            "file_count": len(parsed_files),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "indexed_count": indexed_count,
            "status": "indexed",
            "embedding_model": OLLAMA_EMBED_MODEL,
            "collection_name": CHROMA_COLLECTION_NAME,
        }

    except Exception as exc:
        try:
            update_metadata(
                project_id,
                status="indexing_failed",
                indexing_error=str(exc),
            )

        except Exception:
            # Preserve the original indexing error.
            pass

        raise