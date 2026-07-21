import json
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4
import sys


from .file_scanner import (
    get_ast_path,
    get_chunks_path,
    get_project_directory,
    get_source_directory,  # needed for source file lookup
    update_metadata,
    validate_project_id,
)

LANGUAGE = "python"


def load_ast(project_id: str) -> Dict[str, Any]:
    ast_path = get_ast_path(project_id)

    if not ast_path.exists():
        raise FileNotFoundError(
            f"ast.json not found for project '{project_id}'. "
            "Run the parser before chunking."
        )

    try:
        return json.loads(ast_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ast.json is invalid for project '{project_id}'"
        ) from exc


def read_source_lines(project_id: str, relative_path: str) -> List[str]:
    # Files are stored under .../<project_id>/source/
    file_path = get_source_directory(project_id) / relative_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {file_path}"
        )

    return file_path.read_text(encoding="utf-8").splitlines()


def make_chunk_id(project_id: str) -> str:
    return f"{project_id}_{uuid4().hex[:12]}"


def build_chunk(
    project_id: str,
    file_path: str,
    source_lines: List[str],
    name: str,
    start_line: int,
    end_line: int,
    source_type: str,
    docstring: str = None,
) -> Dict[str, Any]:
    content = "\n".join(source_lines[start_line - 1:end_line])
    return {
        "chunk_id": make_chunk_id(project_id),
        "content": content,
        "metadata": {
            "project_id": project_id,
            "file_path": file_path,
            "language": LANGUAGE,
            "start_line": start_line,
            "end_line": end_line,
            "source_type": source_type,
            "name": name,
            "docstring": docstring,
        },
    }


def chunk_file(
    project_id: str,
    file_entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if file_entry.get("status") != "parsed":
        return []

    relative_path = file_entry["path"]
    source_lines = read_source_lines(project_id, relative_path)

    chunks: List[Dict[str, Any]] = []

    for function in file_entry.get("functions", []):
        chunks.append(
            build_chunk(
                project_id=project_id,
                file_path=relative_path,
                source_lines=source_lines,
                name=function["name"],
                start_line=function["start_line"],
                end_line=function["end_line"],
                source_type="function",
                docstring=function.get("docstring"),
            )
        )

    for class_entry in file_entry.get("classes", []):
        chunks.append(
            build_chunk(
                project_id=project_id,
                file_path=relative_path,
                source_lines=source_lines,
                name=class_entry["name"],
                start_line=class_entry["start_line"],
                end_line=class_entry["end_line"],
                source_type="class",
                docstring=class_entry.get("docstring"),
            )
        )

    return chunks


def chunk_project(project_id: str) -> List[Dict[str, Any]]:
    validate_project_id(project_id)
    ast_data = load_ast(project_id)

    all_chunks: List[Dict[str, Any]] = []

    for file_entry in ast_data.get("files", []):
        all_chunks.extend(chunk_file(project_id, file_entry))

    chunks_path = get_chunks_path(project_id)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(all_chunks, indent=2),
        encoding="utf-8",
    )

    update_metadata(
        project_id,
        chunk_count=len(all_chunks),
        status="chunked",
    )

    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m backend.services.chunker <project_id>")
        sys.exit(1)

    result = chunk_project(sys.argv[1])
    print(f"Created {len(result)} chunks")
    for chunk in result[:3]:
        print(chunk["metadata"])