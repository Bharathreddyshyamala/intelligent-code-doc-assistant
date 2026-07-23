import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from services.indexer import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    get_chroma_collection,
)


def make_json_serializable(value: Any) -> Any:
    """
    Convert NumPy arrays and other iterable objects into
    standard Python values that json.dumps() can process.
    """

    if value is None:
        return None

    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, dict):
        return {
            key: make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    return value


def shorten_document(
    document: Optional[str],
    maximum_length: int = 1000,
) -> str:
    """
    Shorten long documents when displaying them in Terminal.
    """

    if not document:
        return ""

    if len(document) <= maximum_length:
        return document

    return (
        document[:maximum_length]
        + "\n... [document shortened]"
    )


def inspect_chroma(
    project_id: Optional[str] = None,
    limit: int = 100,
    show_full_vectors: bool = False,
    export_json: bool = False,
) -> None:
    """
    Display readable ChromaDB records.

    When project_id is supplied, only records belonging to
    that project are returned.
    """

    collection = get_chroma_collection()

    get_arguments: Dict[str, Any] = {
        "limit": limit,
        "include": [
            "documents",
            "metadatas",
            "embeddings",
        ],
    }

    if project_id:
        get_arguments["where"] = {
            "project_id": project_id,
        }

    results = collection.get(
        **get_arguments
    )

    ids = results.get("ids", [])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    embeddings = results.get("embeddings")

    print("=" * 80)
    print("ChromaDB inspection")
    print("=" * 80)
    print(f"Database path: {CHROMA_DB_PATH}")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Total collection records: {collection.count()}")

    if project_id:
        print(f"Project filter: {project_id}")
    else:
        print("Project filter: all projects")

    print(f"Records returned: {len(ids)}")
    print("=" * 80)

    if not ids:
        print("No matching records were found.")
        return

    exported_records = []

    for index, chunk_id in enumerate(ids):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        document = (
            documents[index]
            if index < len(documents)
            else ""
        )

        embedding = None

        if embeddings is not None and index < len(embeddings):
            embedding = make_json_serializable(
                embeddings[index]
            )

        print()
        print("-" * 80)
        print(f"Record {index + 1}")
        print("-" * 80)

        print(f"Chunk ID: {chunk_id}")
        print(
            "Project ID:",
            metadata.get("project_id", ""),
        )
        print(
            "File:",
            metadata.get("file_path", ""),
        )
        print(
            "Symbol:",
            metadata.get("name", ""),
        )
        print(
            "Type:",
            metadata.get("source_type", ""),
        )
        print(
            "Lines:",
            (
                f"{metadata.get('start_line', '')}-"
                f"{metadata.get('end_line', '')}"
            ),
        )
        print(
            "Embedding model:",
            metadata.get("embedding_model", ""),
        )

        if embedding is not None:
            print(
                "Vector dimensions:",
                len(embedding),
            )

            if show_full_vectors:
                print("Complete vector:")
                print(embedding)
            else:
                print(
                    "First 10 vector values:",
                    embedding[:10],
                )

        print("\nStored document:")
        print(
            shorten_document(document)
        )

        print("\nMetadata:")
        print(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            )
        )

        exported_records.append(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "embedding_dimension": (
                    len(embedding)
                    if embedding is not None
                    else 0
                ),
                "embedding": embedding,
            }
        )

    if export_json:
        output_directory = (
            Path(__file__).resolve().parent
            / "vector_store"
            / "exports"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_name = (
            f"{project_id}_records.json"
            if project_id
            else "all_chroma_records.json"
        )

        output_path = (
            output_directory / output_name
        )

        output_path.write_text(
            json.dumps(
                exported_records,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 80)
        print(f"Records exported to: {output_path}")
        print("=" * 80)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display readable records stored in ChromaDB."
        )
    )

    parser.add_argument(
        "--project-id",
        help=(
            "Only show records belonging to this project."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of records to return.",
    )

    parser.add_argument(
        "--full-vectors",
        action="store_true",
        help=(
            "Print every number in each embedding vector."
        ),
    )

    parser.add_argument(
        "--export-json",
        action="store_true",
        help=(
            "Export the returned records into a readable "
            "JSON file."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    inspect_chroma(
        project_id=arguments.project_id,
        limit=arguments.limit,
        show_full_vectors=arguments.full_vectors,
        export_json=arguments.export_json,
    )