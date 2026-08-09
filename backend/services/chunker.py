import hashlib
import json
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


from .file_scanner import (
    get_ast_path,
    get_chunks_path,
    get_source_directory,
    update_metadata,
    validate_project_id,
)


CHUNKING_VERSION = "2.0"


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "configuration",
    ".conf": "configuration",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".xml": "xml",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
}


SPECIAL_TEXT_FILENAMES = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "procfile": "procfile",
    "requirements.txt": "requirements",
    "pyproject.toml": "toml",
    "package.json": "json",
    "package-lock.json": "json",
    "readme": "markdown",
    "readme.md": "markdown",
}


EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    "target",
    "coverage",
    ".next",
    ".nuxt",
    "vendor",
    "generated_docs",
    "vector_store",
    "chroma_db",
}


SENSITIVE_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "id_rsa",
    "id_ed25519",
}


SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
}


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Configuration for semantic and fallback chunking.

    max_chars:
        Approximate maximum chunk size.

    max_lines:
        Maximum lines in one code chunk.

    min_lines:
        Minimum target size before searching for a natural
        blank-line boundary.

    overlap_lines:
        Lines repeated between neighboring chunks.

    include_module_summary:
        Create a searchable module overview.

    include_top_level_code:
        Include imports, constants, route setup, application setup,
        and other code outside classes and functions.

    include_unparsed_text_files:
        Chunk supported text files even when AST metadata is unavailable.

    reuse_unchanged_files:
        Reuse existing chunks when the file content and configuration
        have not changed.

    max_file_bytes:
        Avoid processing extremely large generated or data files.
    """

    max_chars: int = 6_000
    max_lines: int = 120
    min_lines: int = 20
    overlap_lines: int = 12

    include_module_summary: bool = True
    include_top_level_code: bool = True
    include_unparsed_text_files: bool = True
    reuse_unchanged_files: bool = True

    max_file_bytes: int = 1_000_000
    maximum_docstring_characters: int = 1_200

    def validate(self) -> None:
        if self.max_chars < 500:
            raise ValueError(
                "max_chars must be at least 500."
            )

        if self.max_lines < 10:
            raise ValueError(
                "max_lines must be at least 10."
            )

        if self.min_lines < 1:
            raise ValueError(
                "min_lines must be greater than zero."
            )

        if self.min_lines > self.max_lines:
            raise ValueError(
                "min_lines cannot be greater than max_lines."
            )

        if self.overlap_lines < 0:
            raise ValueError(
                "overlap_lines cannot be negative."
            )

        if self.overlap_lines >= self.max_lines:
            raise ValueError(
                "overlap_lines must be smaller than max_lines."
            )


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


@dataclass(frozen=True)
class SourceSegment:
    """
    A source-code segment with its original line numbers.
    """

    start_line: int
    end_line: int
    text: str


def load_ast(project_id: str) -> Dict[str, Any]:
    """
    Load the project's ast.json.
    """

    ast_path = get_ast_path(project_id)

    if not ast_path.exists():
        raise FileNotFoundError(
            f"ast.json not found for project '{project_id}'. "
            "Run the parser before chunking."
        )

    try:
        ast_data = json.loads(
            ast_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ast.json is invalid for project "
            f"'{project_id}'."
        ) from exc

    if not isinstance(ast_data, dict):
        raise ValueError(
            "ast.json must contain a JSON object."
        )

    return ast_data


def _safe_source_path(
    project_id: str,
    relative_path: str,
) -> Path:
    """
    Resolve a path while preventing access outside the project source
    directory.
    """

    source_directory = get_source_directory(
        project_id
    ).resolve()

    requested_path = (
        source_directory / relative_path
    ).resolve()

    if (
        requested_path != source_directory
        and source_directory
        not in requested_path.parents
    ):
        raise ValueError(
            f"Unsafe source path: {relative_path}"
        )

    return requested_path


def read_source_lines(
    project_id: str,
    relative_path: str,
) -> List[str]:
    """
    Read a source file as lines.

    Invalid characters are replaced rather than causing the complete
    project chunking process to fail.
    """

    file_path = _safe_source_path(
        project_id,
        relative_path,
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {file_path}"
        )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Source path is not a file: {file_path}"
        )

    return file_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()


def _hash_text(text: str) -> str:
    """
    Create a SHA-256 content hash.
    """

    return hashlib.sha256(
        text.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()


def _configuration_signature(
    config: ChunkingConfig,
) -> str:
    """
    Create a signature representing the active chunking configuration.

    Existing chunks are reused only when this value matches.
    """

    config_json = json.dumps(
        asdict(config),
        sort_keys=True,
    )

    return _hash_text(config_json)[:16]


def make_chunk_id(
    project_id: str,
    semantic_key: str,
) -> str:
    """
    Create a stable chunk ID.

    Unlike UUID-based IDs, this ID remains the same when the same
    symbol is reprocessed.

    Stable IDs allow ChromaDB upsert operations and prevent unnecessary
    duplicate vector records.
    """

    digest = hashlib.sha256(
        semantic_key.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()[:20]

    return f"{project_id}_{digest}"


def estimate_token_count(text: str) -> int:
    """
    Estimate token count without requiring an external tokenizer.

    Source code commonly averages approximately three to four
    characters per token. This is only used for metadata and context
    budgeting.
    """

    if not text:
        return 0

    return max(
        1,
        (len(text) + 3) // 4,
    )


def detect_language(
    file_path: str,
) -> str:
    """
    Detect a language from the file name or extension.
    """

    path = Path(file_path)
    lower_name = path.name.lower()

    if lower_name in SPECIAL_TEXT_FILENAMES:
        return SPECIAL_TEXT_FILENAMES[
            lower_name
        ]

    return LANGUAGE_BY_SUFFIX.get(
        path.suffix.lower(),
        "text",
    )


def _normalise_text(
    value: Any,
    maximum_characters: Optional[int] = None,
) -> str:
    """
    Convert AST values into safe text.
    """

    if value is None:
        text = ""

    elif isinstance(value, str):
        text = value.strip()

    elif isinstance(value, (int, float, bool)):
        text = str(value)

    else:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
            )

        except TypeError:
            text = str(value)

    if (
        maximum_characters is not None
        and len(text) > maximum_characters
    ):
        text = (
            text[:maximum_characters]
            + "..."
        )

    return text


def _metadata_scalar(
    value: Any,
) -> Any:
    """
    Convert metadata to values accepted by ChromaDB.

    Chroma metadata should contain strings, integers, floats, or booleans,
    not nested dictionaries or lists.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return _normalise_text(value)


def _format_parameter(
    parameter: Any,
) -> str:
    """
    Convert a parameter AST object into searchable text.
    """

    if isinstance(parameter, str):
        return parameter

    if not isinstance(parameter, dict):
        return _normalise_text(parameter)

    name = (
        parameter.get("name")
        or parameter.get("arg")
        or "parameter"
    )

    annotation = (
        parameter.get("annotation")
        or parameter.get("type")
        or ""
    )

    default = parameter.get("default")

    output = str(name)

    if annotation:
        output += f": {annotation}"

    if default not in (None, ""):
        output += f" = {default}"

    return output


def _format_parameters(
    symbol: Dict[str, Any],
) -> str:
    """
    Extract function or method parameters from different AST schemas.
    """

    parameters = (
        symbol.get("parameters")
        or symbol.get("args")
        or symbol.get("arguments")
        or []
    )

    if isinstance(parameters, dict):
        parameter_values: Iterable[Any] = (
            parameters.values()
        )

    elif isinstance(parameters, list):
        parameter_values = parameters

    else:
        return _normalise_text(parameters)

    return ", ".join(
        _format_parameter(parameter)
        for parameter in parameter_values
    )


def _format_decorators(
    symbol: Dict[str, Any],
) -> str:
    """
    Extract decorators from a class, method, or function.
    """

    decorators = (
        symbol.get("decorators")
        or symbol.get("decorator_list")
        or []
    )

    if not isinstance(decorators, list):
        return _normalise_text(decorators)

    return ", ".join(
        _normalise_text(decorator)
        for decorator in decorators
        if _normalise_text(decorator)
    )


def _format_import(
    import_entry: Any,
) -> str:
    """
    Convert different import schemas to readable text.
    """

    if isinstance(import_entry, str):
        return import_entry

    if not isinstance(import_entry, dict):
        return _normalise_text(import_entry)

    module = (
        import_entry.get("module")
        or import_entry.get("name")
        or ""
    )

    imported_names = (
        import_entry.get("names")
        or import_entry.get("imports")
        or []
    )

    alias = import_entry.get("alias")

    if isinstance(imported_names, list):
        imported_text = ", ".join(
            _normalise_text(item)
            for item in imported_names
        )

    else:
        imported_text = _normalise_text(
            imported_names
        )

    if module and imported_text:
        text = (
            f"from {module} import "
            f"{imported_text}"
        )

    elif module:
        text = f"import {module}"

    else:
        text = imported_text

    if alias:
        text += f" as {alias}"

    return text


def _extract_imports(
    file_entry: Dict[str, Any],
) -> List[str]:
    """
    Extract readable imports from a file AST record.
    """

    imports = file_entry.get(
        "imports",
        [],
    )

    if not isinstance(imports, list):
        imports = [imports]

    return [
        formatted
        for item in imports
        if (
            formatted := _format_import(item)
        )
    ]


def _get_symbol_name(
    symbol: Dict[str, Any],
    default: str,
) -> str:
    """
    Return a safe symbol name.
    """

    name = symbol.get("name")

    if isinstance(name, str) and name.strip():
        return name.strip()

    return default


def _valid_line_range(
    start_line: Any,
    end_line: Any,
    total_lines: int,
) -> Optional[Tuple[int, int]]:
    """
    Validate and clamp AST line numbers.
    """

    try:
        start = int(start_line)
        end = int(end_line)

    except (TypeError, ValueError):
        return None

    start = max(
        1,
        min(start, total_lines),
    )

    end = max(
        start,
        min(end, total_lines),
    )

    return start, end


def _split_source_region(
    source_lines: Sequence[str],
    start_line: int,
    end_line: int,
    config: ChunkingConfig,
) -> List[SourceSegment]:
    """
    Divide a source region using line and character limits.

    Natural blank-line boundaries are preferred. Neighboring segments
    overlap so that a continuation chunk keeps surrounding context.
    """

    if not source_lines:
        return []

    total_lines = len(source_lines)

    validated_range = _valid_line_range(
        start_line,
        end_line,
        total_lines,
    )

    if validated_range is None:
        return []

    region_start, region_end = (
        validated_range
    )

    segments: List[SourceSegment] = []
    cursor = region_start

    while cursor <= region_end:
        current_line = source_lines[
            cursor - 1
        ]

        # Handle minified source or generated text containing one
        # exceptionally large line.
        if len(current_line) > config.max_chars:
            character_overlap = min(
                300,
                max(
                    0,
                    config.max_chars // 10,
                ),
            )

            character_start = 0

            while character_start < len(
                current_line
            ):
                character_end = min(
                    len(current_line),
                    character_start
                    + config.max_chars,
                )

                segments.append(
                    SourceSegment(
                        start_line=cursor,
                        end_line=cursor,
                        text=current_line[
                            character_start:
                            character_end
                        ],
                    )
                )

                if character_end >= len(
                    current_line
                ):
                    break

                character_start = max(
                    character_start + 1,
                    character_end
                    - character_overlap,
                )

            cursor += 1
            continue

        candidate_end = cursor - 1
        character_count = 0

        while candidate_end < region_end:
            next_line = source_lines[
                candidate_end
            ]

            next_size = len(next_line) + 1
            current_line_count = (
                candidate_end
                - cursor
                + 1
            )

            if (
                current_line_count
                >= config.max_lines
            ):
                break

            if (
                candidate_end >= cursor
                and character_count
                + next_size
                > config.max_chars
            ):
                break

            character_count += next_size
            candidate_end += 1

        if candidate_end < cursor:
            candidate_end = cursor

        selected_end = candidate_end

        # Prefer ending at a blank line rather than splitting in the middle of a logical section.
        
        if selected_end < region_end:
            minimum_boundary = min(
                selected_end,
                cursor
                + config.min_lines
                - 1,
            )

            for line_number in range(
                selected_end,
                minimum_boundary - 1,
                -1,
            ):
                if not source_lines[
                    line_number - 1
                ].strip():
                    selected_end = line_number
                    break

        segment_text = "\n".join(
            source_lines[
                cursor - 1:
                selected_end
            ]
        )

        segments.append(
            SourceSegment(
                start_line=cursor,
                end_line=selected_end,
                text=segment_text,
            )
        )

        if selected_end >= region_end:
            break

        cursor = max(
            cursor + 1,
            selected_end
            - config.overlap_lines
            + 1,
        )

    return segments


def _render_contextual_content(
    *,
    raw_content: str,
    file_path: str,
    language: str,
    source_type: str,
    name: str,
    qualified_name: str,
    parent_name: str,
    start_line: int,
    end_line: int,
    part_index: int,
    part_count: int,
    docstring: str,
    chunk_kind: str,
) -> str:
    """
    Add compact semantic context before the source.

    This improves both dense embeddings and sparse BM25 retrieval because
    the symbol name, file, parent class, and type are present in the text
    being indexed.
    """

    context_lines = [
        "[Code Chunk Context]",
        f"File: {file_path}",
        f"Language: {language}",
        f"Type: {source_type}",
        f"Chunk kind: {chunk_kind}",
        f"Symbol: {name}",
        f"Qualified symbol: {qualified_name}",
        (
            f"Lines: {start_line}-{end_line}"
        ),
        (
            f"Part: {part_index}/{part_count}"
        ),
    ]

    if parent_name:
        context_lines.append(
            f"Parent: {parent_name}"
        )

    if docstring:
        context_lines.append(
            "Docstring: "
            + docstring[:500].replace(
                "\n",
                " ",
            )
        )

    return (
        "\n".join(context_lines)
        + "\n\n[Source Content]\n"
        + raw_content
    ).strip()


def _create_chunk_record(
    *,
    project_id: str,
    file_path: str,
    raw_content: str,
    name: str,
    qualified_name: str,
    start_line: int,
    end_line: int,
    source_type: str,
    docstring: str,
    language: str,
    parent_name: str,
    part_index: int,
    part_count: int,
    chunk_kind: str,
    file_hash: str,
    config_signature: str,
    parser_status: str,
    extra_metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Build the final chunk record.

    Existing keys are preserved for backward compatibility.
    """

    semantic_key = "|".join(
        [
            file_path,
            source_type,
            qualified_name,
            chunk_kind,
            str(part_index),
        ]
    )

    chunk_id = make_chunk_id(
        project_id,
        semantic_key,
    )

    contextual_content = (
        _render_contextual_content(
            raw_content=raw_content,
            file_path=file_path,
            language=language,
            source_type=source_type,
            name=name,
            qualified_name=qualified_name,
            parent_name=parent_name,
            start_line=start_line,
            end_line=end_line,
            part_index=part_index,
            part_count=part_count,
            docstring=docstring,
            chunk_kind=chunk_kind,
        )
    )

    metadata: Dict[str, Any] = {
        # Existing metadata fields
        "project_id": project_id,
        "file_path": file_path,
        "language": language,
        "start_line": start_line,
        "end_line": end_line,
        "source_type": source_type,
        "name": name,
        "docstring": docstring,

        # New backward-compatible metadata
        "qualified_name": qualified_name,
        "parent_name": parent_name,
        "chunk_kind": chunk_kind,
        "chunk_index": part_index,
        "chunk_count": part_count,
        "is_partial": part_count > 1,
        "token_estimate": (
            estimate_token_count(
                contextual_content
            )
        ),
        "content_hash": _hash_text(
            raw_content
        ),
        "file_hash": file_hash,
        "parser_status": parser_status,
        "chunking_version": (
            CHUNKING_VERSION
        ),
        "chunking_signature": (
            config_signature
        ),
    }

    if extra_metadata:
        for key, value in (
            extra_metadata.items()
        ):
            metadata[key] = (
                _metadata_scalar(value)
            )

    return {
        "chunk_id": chunk_id,

        # Existing indexers can continue embedding this value.
        "content": contextual_content,

        # Useful for documentation generation and source display.
        "raw_content": raw_content,

        "metadata": metadata,
    }


def build_chunk(
    project_id: str,
    file_path: str,
    source_lines: List[str],
    name: str,
    start_line: int,
    end_line: int,
    source_type: str,
    docstring: Optional[str] = None,
    *,
    parent_name: str = "",
    qualified_name: Optional[str] = None,
    part_index: int = 1,
    part_count: int = 1,
    language: Optional[str] = None,
    chunk_kind: str = "code",
    file_hash: str = "",
    config_signature: str = "",
    parser_status: str = "parsed",
    extra_metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible chunk builder.

    The original positional arguments remain valid.
    """

    validated_range = _valid_line_range(
        start_line,
        end_line,
        len(source_lines),
    )

    if validated_range is None:
        raise ValueError(
            f"Invalid line range for {name}: "
            f"{start_line}-{end_line}"
        )

    valid_start, valid_end = (
        validated_range
    )

    raw_content = "\n".join(
        source_lines[
            valid_start - 1:
            valid_end
        ]
    )

    detected_language = (
        language
        or detect_language(file_path)
    )

    final_qualified_name = (
        qualified_name
        or (
            f"{parent_name}.{name}"
            if parent_name
            else name
        )
    )

    final_file_hash = (
        file_hash
        or _hash_text(
            "\n".join(source_lines)
        )
    )

    final_signature = (
        config_signature
        or _configuration_signature(
            DEFAULT_CHUNKING_CONFIG
        )
    )

    return _create_chunk_record(
        project_id=project_id,
        file_path=file_path,
        raw_content=raw_content,
        name=name,
        qualified_name=(
            final_qualified_name
        ),
        start_line=valid_start,
        end_line=valid_end,
        source_type=source_type,
        docstring=_normalise_text(
            docstring,
            DEFAULT_CHUNKING_CONFIG
            .maximum_docstring_characters,
        ),
        language=detected_language,
        parent_name=parent_name,
        part_index=part_index,
        part_count=part_count,
        chunk_kind=chunk_kind,
        file_hash=final_file_hash,
        config_signature=final_signature,
        parser_status=parser_status,
        extra_metadata=extra_metadata,
    )


def _create_segment_chunks(
    *,
    project_id: str,
    file_path: str,
    segments: List[SourceSegment],
    name: str,
    qualified_name: str,
    source_type: str,
    docstring: str,
    language: str,
    parent_name: str,
    chunk_kind: str,
    file_hash: str,
    config_signature: str,
    parser_status: str,
    extra_metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> List[Dict[str, Any]]:
    """
    Convert source segments into final chunk records.
    """

    part_count = len(segments)

    return [
        _create_chunk_record(
            project_id=project_id,
            file_path=file_path,
            raw_content=segment.text,
            name=name,
            qualified_name=qualified_name,
            start_line=segment.start_line,
            end_line=segment.end_line,
            source_type=source_type,
            docstring=docstring,
            language=language,
            parent_name=parent_name,
            part_index=part_index,
            part_count=part_count,
            chunk_kind=chunk_kind,
            file_hash=file_hash,
            config_signature=(
                config_signature
            ),
            parser_status=parser_status,
            extra_metadata=extra_metadata,
        )
        for part_index, segment in enumerate(
            segments,
            start=1,
        )
    ]


def _chunk_symbol(
    *,
    project_id: str,
    file_path: str,
    source_lines: List[str],
    symbol: Dict[str, Any],
    source_type: str,
    parent_name: str,
    config: ChunkingConfig,
    file_hash: str,
    config_signature: str,
    parser_status: str,
) -> List[Dict[str, Any]]:
    """
    Chunk a function or method using its AST boundaries.
    """

    symbol_name = _get_symbol_name(
        symbol,
        source_type,
    )

    qualified_name = (
        _normalise_text(
            symbol.get("qualified_name")
        )
        or (
            f"{parent_name}.{symbol_name}"
            if parent_name
            else symbol_name
        )
    )

    line_range = _valid_line_range(
        symbol.get("start_line"),
        symbol.get("end_line"),
        len(source_lines),
    )

    if line_range is None:
        return []

    start_line, end_line = line_range

    segments = _split_source_region(
        source_lines=source_lines,
        start_line=start_line,
        end_line=end_line,
        config=config,
    )

    if not segments:
        return []

    docstring = _normalise_text(
        symbol.get("docstring"),
        config.maximum_docstring_characters,
    )

    return_annotation = (
        symbol.get("return_annotation")
        or symbol.get("returns")
        or symbol.get("return_type")
        or ""
    )

    extra_metadata = {
        "parameters": _format_parameters(
            symbol
        ),
        "return_annotation": (
            _normalise_text(
                return_annotation
            )
        ),
        "decorators": (
            _format_decorators(symbol)
        ),
        "is_async": bool(
            symbol.get("is_async", False)
        ),
    }

    return _create_segment_chunks(
        project_id=project_id,
        file_path=file_path,
        segments=segments,
        name=symbol_name,
        qualified_name=qualified_name,
        source_type=source_type,
        docstring=docstring,
        language=detect_language(
            file_path
        ),
        parent_name=parent_name,
        chunk_kind="symbol_code",
        file_hash=file_hash,
        config_signature=config_signature,
        parser_status=parser_status,
        extra_metadata=extra_metadata,
    )


def _get_class_methods(
    class_entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Read methods from common AST schemas.
    """

    methods = (
        class_entry.get("methods")
        or class_entry.get("functions")
        or []
    )

    if not isinstance(methods, list):
        return []

    return [
        method
        for method in methods
        if isinstance(method, dict)
    ]


def _format_names(
    values: Any,
) -> str:
    """
    Convert names, bases, attributes, or similar AST lists to text.
    """

    if values is None:
        return ""

    if not isinstance(values, list):
        return _normalise_text(values)

    output: List[str] = []

    for value in values:
        if isinstance(value, dict):
            display_value = (
                value.get("name")
                or value.get("id")
                or value.get("value")
                or _normalise_text(value)
            )

        else:
            display_value = (
                _normalise_text(value)
            )

        if display_value:
            output.append(
                str(display_value)
            )

    return ", ".join(output)


def _create_class_summary(
    *,
    project_id: str,
    file_path: str,
    class_entry: Dict[str, Any],
    source_lines: List[str],
    file_hash: str,
    config_signature: str,
    parser_status: str,
    config: ChunkingConfig,
) -> Optional[Dict[str, Any]]:
    """
    Build a compact class overview without duplicating all method bodies.
    """

    class_name = _get_symbol_name(
        class_entry,
        "class",
    )

    line_range = _valid_line_range(
        class_entry.get("start_line"),
        class_entry.get("end_line"),
        len(source_lines),
    )

    if line_range is None:
        return None

    start_line, end_line = line_range

    methods = _get_class_methods(
        class_entry
    )

    method_names = [
        _get_symbol_name(
            method,
            "method",
        )
        for method in methods
    ]

    bases = (
        class_entry.get("bases")
        or class_entry.get("base_classes")
        or []
    )

    attributes = (
        class_entry.get("attributes")
        or class_entry.get(
            "class_attributes"
        )
        or []
    )

    docstring = _normalise_text(
        class_entry.get("docstring"),
        config.maximum_docstring_characters,
    )

    summary_lines = [
        f"Class: {class_name}",
        f"File: {file_path}",
        (
            f"Lines: {start_line}-"
            f"{end_line}"
        ),
    ]

    formatted_bases = _format_names(bases)

    if formatted_bases:
        summary_lines.append(
            f"Base classes: "
            f"{formatted_bases}"
        )

    if docstring:
        summary_lines.extend(
            [
                "Docstring:",
                docstring,
            ]
        )

    if method_names:
        summary_lines.append(
            "Methods: "
            + ", ".join(method_names)
        )

    formatted_attributes = _format_names(
        attributes
    )

    if formatted_attributes:
        summary_lines.append(
            "Attributes: "
            + formatted_attributes
        )

    decorators = _format_decorators(
        class_entry
    )

    if decorators:
        summary_lines.append(
            f"Decorators: {decorators}"
        )

    summary_text = "\n".join(
        summary_lines
    )

    return _create_chunk_record(
        project_id=project_id,
        file_path=file_path,
        raw_content=summary_text,
        name=class_name,
        qualified_name=class_name,
        start_line=start_line,
        end_line=end_line,
        source_type="class",
        docstring=docstring,
        language=detect_language(
            file_path
        ),
        parent_name="",
        part_index=1,
        part_count=1,
        chunk_kind="class_summary",
        file_hash=file_hash,
        config_signature=config_signature,
        parser_status=parser_status,
        extra_metadata={
            "methods": ", ".join(
                method_names
            ),
            "bases": formatted_bases,
            "attributes": (
                formatted_attributes
            ),
            "decorators": decorators,
        },
    )


def _merge_intervals(
    intervals: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Merge overlapping source line intervals.
    """

    if not intervals:
        return []

    sorted_intervals = sorted(
        intervals,
        key=lambda interval: (
            interval[0],
            interval[1],
        ),
    )

    merged: List[Tuple[int, int]] = [
        sorted_intervals[0]
    ]

    for current_start, current_end in (
        sorted_intervals[1:]
    ):
        previous_start, previous_end = (
            merged[-1]
        )

        if current_start <= previous_end + 1:
            merged[-1] = (
                previous_start,
                max(
                    previous_end,
                    current_end,
                ),
            )

        else:
            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

    return merged


def _uncovered_intervals(
    start_line: int,
    end_line: int,
    covered_intervals: List[
        Tuple[int, int]
    ],
) -> List[Tuple[int, int]]:
    """
    Return line regions not covered by symbols.

    This captures imports, constants, decorators, route registration,
    application setup, and class-level assignments.
    """

    if start_line > end_line:
        return []

    relevant_intervals = []

    for covered_start, covered_end in (
        covered_intervals
    ):
        if covered_end < start_line:
            continue

        if covered_start > end_line:
            continue

        relevant_intervals.append(
            (
                max(
                    start_line,
                    covered_start,
                ),
                min(
                    end_line,
                    covered_end,
                ),
            )
        )

    merged = _merge_intervals(
        relevant_intervals
    )

    uncovered: List[Tuple[int, int]] = []
    cursor = start_line

    for covered_start, covered_end in merged:
        if cursor < covered_start:
            uncovered.append(
                (
                    cursor,
                    covered_start - 1,
                )
            )

        cursor = max(
            cursor,
            covered_end + 1,
        )

    if cursor <= end_line:
        uncovered.append(
            (
                cursor,
                end_line,
            )
        )

    return uncovered


def _contains_meaningful_code(
    source_lines: Sequence[str],
    start_line: int,
    end_line: int,
) -> bool:
    """
    Ignore ranges containing only whitespace.
    """

    return any(
        line.strip()
        for line in source_lines[
            start_line - 1:
            end_line
        ]
    )


def _create_module_summary(
    *,
    project_id: str,
    file_entry: Dict[str, Any],
    source_lines: List[str],
    file_hash: str,
    config_signature: str,
    parser_status: str,
    config: ChunkingConfig,
) -> Optional[Dict[str, Any]]:
    """
    Create a searchable file-level overview.
    """

    if not source_lines:
        return None

    file_path = file_entry["path"]

    module_name = (
        file_entry.get("module_name")
        or Path(file_path).stem
    )

    functions = [
        _get_symbol_name(
            function,
            "function",
        )
        for function in file_entry.get(
            "functions",
            [],
        )
        if isinstance(function, dict)
    ]

    classes = [
        _get_symbol_name(
            class_entry,
            "class",
        )
        for class_entry in file_entry.get(
            "classes",
            [],
        )
        if isinstance(class_entry, dict)
    ]

    imports = _extract_imports(
        file_entry
    )

    module_docstring = _normalise_text(
        file_entry.get(
            "module_docstring"
        ),
        config.maximum_docstring_characters,
    )

    summary_lines = [
        f"Module: {module_name}",
        f"File: {file_path}",
        (
            f"Language: "
            f"{detect_language(file_path)}"
        ),
    ]

    if module_docstring:
        summary_lines.extend(
            [
                "Module docstring:",
                module_docstring,
            ]
        )

    if imports:
        summary_lines.append(
            "Imports: "
            + ", ".join(imports)
        )

    if classes:
        summary_lines.append(
            "Classes: "
            + ", ".join(classes)
        )

    if functions:
        summary_lines.append(
            "Functions: "
            + ", ".join(functions)
        )

    return _create_chunk_record(
        project_id=project_id,
        file_path=file_path,
        raw_content="\n".join(
            summary_lines
        ),
        name=str(module_name),
        qualified_name=str(module_name),
        start_line=1,
        end_line=len(source_lines),
        source_type="module",
        docstring=module_docstring,
        language=detect_language(
            file_path
        ),
        parent_name="",
        part_index=1,
        part_count=1,
        chunk_kind="module_summary",
        file_hash=file_hash,
        config_signature=config_signature,
        parser_status=parser_status,
        extra_metadata={
            "imports": ", ".join(
                imports
            ),
            "classes": ", ".join(
                classes
            ),
            "functions": ", ".join(
                functions
            ),
        },
    )


def _classify_top_level_functions(
    file_entry: Dict[str, Any],
    class_intervals: List[
        Tuple[int, int]
    ],
) -> List[Dict[str, Any]]:
    """
    Remove function records that are actually methods inside classes.

    Some AST formats include methods in both `functions` and
    `classes[].methods`.
    """

    top_level_functions: List[
        Dict[str, Any]
    ] = []

    for function in file_entry.get(
        "functions",
        [],
    ):
        if not isinstance(function, dict):
            continue

        try:
            start_line = int(
                function.get("start_line")
            )

        except (TypeError, ValueError):
            top_level_functions.append(
                function
            )
            continue

        is_inside_class = any(
            class_start
            <= start_line
            <= class_end
            for class_start, class_end
            in class_intervals
        )

        if not is_inside_class:
            top_level_functions.append(
                function
            )

    return top_level_functions


def _chunk_parsed_file(
    project_id: str,
    file_entry: Dict[str, Any],
    source_lines: List[str],
    config: ChunkingConfig,
    file_hash: str,
    config_signature: str,
) -> List[Dict[str, Any]]:
    """
    AST-aware hierarchical chunking for a parsed file.
    """

    relative_path = file_entry["path"]
    parser_status = file_entry.get(
        "status",
        "parsed",
    )

    chunks: List[Dict[str, Any]] = []
    total_lines = len(source_lines)

    if total_lines == 0:
        return []

    if config.include_module_summary:
        module_summary = (
            _create_module_summary(
                project_id=project_id,
                file_entry=file_entry,
                source_lines=source_lines,
                file_hash=file_hash,
                config_signature=(
                    config_signature
                ),
                parser_status=parser_status,
                config=config,
            )
        )

        if module_summary:
            chunks.append(
                module_summary
            )

    class_entries = [
        item
        for item in file_entry.get(
            "classes",
            [],
        )
        if isinstance(item, dict)
    ]

    class_intervals: List[
        Tuple[int, int]
    ] = []

    for class_entry in class_entries:
        class_range = _valid_line_range(
            class_entry.get("start_line"),
            class_entry.get("end_line"),
            total_lines,
        )

        if class_range:
            class_intervals.append(
                class_range
            )

    top_level_functions = (
        _classify_top_level_functions(
            file_entry,
            class_intervals,
        )
    )

    top_level_symbol_intervals: List[
        Tuple[int, int]
    ] = list(class_intervals)

    # Top-level functions
    for function in top_level_functions:
        function_range = _valid_line_range(
            function.get("start_line"),
            function.get("end_line"),
            total_lines,
        )

        if function_range:
            top_level_symbol_intervals.append(
                function_range
            )

        chunks.extend(
            _chunk_symbol(
                project_id=project_id,
                file_path=relative_path,
                source_lines=source_lines,
                symbol=function,
                source_type="function",
                parent_name="",
                config=config,
                file_hash=file_hash,
                config_signature=(
                    config_signature
                ),
                parser_status=parser_status,
            )
        )

    # Classes and methods
    for class_entry in class_entries:
        class_name = _get_symbol_name(
            class_entry,
            "class",
        )

        class_range = _valid_line_range(
            class_entry.get("start_line"),
            class_entry.get("end_line"),
            total_lines,
        )

        if class_range is None:
            continue

        class_start, class_end = (
            class_range
        )

        class_summary = (
            _create_class_summary(
                project_id=project_id,
                file_path=relative_path,
                class_entry=class_entry,
                source_lines=source_lines,
                file_hash=file_hash,
                config_signature=(
                    config_signature
                ),
                parser_status=parser_status,
                config=config,
            )
        )

        if class_summary:
            chunks.append(
                class_summary
            )

        methods = _get_class_methods(
            class_entry
        )

        method_intervals: List[
            Tuple[int, int]
        ] = []

        for method in methods:
            method_range = _valid_line_range(
                method.get("start_line"),
                method.get("end_line"),
                total_lines,
            )

            if method_range:
                method_intervals.append(
                    method_range
                )

            chunks.extend(
                _chunk_symbol(
                    project_id=project_id,
                    file_path=relative_path,
                    source_lines=source_lines,
                    symbol=method,
                    source_type="method",
                    parent_name=class_name,
                    config=config,
                    file_hash=file_hash,
                    config_signature=(
                        config_signature
                    ),
                    parser_status=(
                        parser_status
                    ),
                )
            )

        # Preserve class declaration, docstring, class variables,
        # attributes, and statements outside method bodies.
        class_context_ranges = (
            _uncovered_intervals(
                class_start,
                class_end,
                method_intervals,
            )
        )

        class_context_segments: List[
            SourceSegment
        ] = []

        for context_start, context_end in (
            class_context_ranges
        ):
            if not _contains_meaningful_code(
                source_lines,
                context_start,
                context_end,
            ):
                continue

            class_context_segments.extend(
                _split_source_region(
                    source_lines=source_lines,
                    start_line=context_start,
                    end_line=context_end,
                    config=config,
                )
            )

        if class_context_segments:
            chunks.extend(
                _create_segment_chunks(
                    project_id=project_id,
                    file_path=relative_path,
                    segments=(
                        class_context_segments
                    ),
                    name=class_name,
                    qualified_name=class_name,
                    source_type="class",
                    docstring=(
                        _normalise_text(
                            class_entry.get(
                                "docstring"
                            ),
                            config
                            .maximum_docstring_characters,
                        )
                    ),
                    language=detect_language(
                        relative_path
                    ),
                    parent_name="",
                    chunk_kind=(
                        "class_context"
                    ),
                    file_hash=file_hash,
                    config_signature=(
                        config_signature
                    ),
                    parser_status=(
                        parser_status
                    ),
                    extra_metadata={
                        "methods": ", ".join(
                            _get_symbol_name(
                                method,
                                "method",
                            )
                            for method in methods
                        ),
                    },
                )
            )

    # Imports, constants, FastAPI router setup, application setup,
    # module-level calls, and other code outside symbols.
    if config.include_top_level_code:
        top_level_ranges = (
            _uncovered_intervals(
                1,
                total_lines,
                top_level_symbol_intervals,
            )
        )

        top_level_segments: List[
            SourceSegment
        ] = []

        for range_start, range_end in (
            top_level_ranges
        ):
            if not _contains_meaningful_code(
                source_lines,
                range_start,
                range_end,
            ):
                continue

            top_level_segments.extend(
                _split_source_region(
                    source_lines=source_lines,
                    start_line=range_start,
                    end_line=range_end,
                    config=config,
                )
            )

        if top_level_segments:
            module_name = (
                file_entry.get("module_name")
                or Path(
                    relative_path
                ).stem
            )

            chunks.extend(
                _create_segment_chunks(
                    project_id=project_id,
                    file_path=relative_path,
                    segments=top_level_segments,
                    name=str(module_name),
                    qualified_name=str(
                        module_name
                    ),
                    source_type="module",
                    docstring=(
                        _normalise_text(
                            file_entry.get(
                                "module_docstring"
                            ),
                            config
                            .maximum_docstring_characters,
                        )
                    ),
                    language=detect_language(
                        relative_path
                    ),
                    parent_name="",
                    chunk_kind=(
                        "top_level_code"
                    ),
                    file_hash=file_hash,
                    config_signature=(
                        config_signature
                    ),
                    parser_status=(
                        parser_status
                    ),
                    extra_metadata={
                        "imports": ", ".join(
                            _extract_imports(
                                file_entry
                            )
                        ),
                    },
                )
            )

    return chunks


def _chunk_plain_text_file(
    project_id: str,
    file_entry: Dict[str, Any],
    source_lines: List[str],
    config: ChunkingConfig,
    file_hash: str,
    config_signature: str,
) -> List[Dict[str, Any]]:
    """
    Fallback chunker for files without AST support.

    This allows configuration, frontend, documentation, SQL, Java,
    JavaScript, and other project files to participate in retrieval.
    """

    if not source_lines:
        return []

    relative_path = file_entry["path"]
    parser_status = file_entry.get(
        "status",
        "unparsed",
    )

    segments = _split_source_region(
        source_lines=source_lines,
        start_line=1,
        end_line=len(source_lines),
        config=config,
    )

    return _create_segment_chunks(
        project_id=project_id,
        file_path=relative_path,
        segments=segments,
        name=Path(relative_path).name,
        qualified_name=relative_path,
        source_type="file",
        docstring="",
        language=detect_language(
            relative_path
        ),
        parent_name="",
        chunk_kind="fallback_text",
        file_hash=file_hash,
        config_signature=config_signature,
        parser_status=parser_status,
    )


def _is_supported_text_file(
    file_path: Path,
) -> bool:
    """
    Check whether a file can safely be treated as project text.
    """

    lower_name = file_path.name.lower()

    if lower_name in SENSITIVE_FILENAMES:
        return False

    if (
        file_path.suffix.lower()
        in SENSITIVE_SUFFIXES
    ):
        return False

    if lower_name in SPECIAL_TEXT_FILENAMES:
        return True

    return (
        file_path.suffix.lower()
        in LANGUAGE_BY_SUFFIX
    )


def _is_excluded_path(
    relative_path: Path,
) -> bool:
    """
    Exclude dependencies, generated outputs, caches, and secrets.
    """

    return any(
        part in EXCLUDED_DIRECTORIES
        for part in relative_path.parts
    )


def _discover_additional_text_files(
    project_id: str,
    known_paths: set[str],
    config: ChunkingConfig,
) -> List[Dict[str, Any]]:
    """
    Discover useful project files not represented by the AST parser.
    """

    if not config.include_unparsed_text_files:
        return []

    source_directory = get_source_directory(
        project_id
    )

    discovered_entries: List[
        Dict[str, Any]
    ] = []

    for file_path in source_directory.rglob(
        "*"
    ):
        if not file_path.is_file():
            continue

        relative_path_object = (
            file_path.relative_to(
                source_directory
            )
        )

        relative_path = (
            relative_path_object.as_posix()
        )

        if relative_path in known_paths:
            continue

        if _is_excluded_path(
            relative_path_object
        ):
            continue

        if not _is_supported_text_file(
            file_path
        ):
            continue

        try:
            file_size = file_path.stat().st_size

        except OSError:
            continue

        if file_size > config.max_file_bytes:
            continue

        discovered_entries.append(
            {
                "path": relative_path,
                "status": "unparsed",
                "language": detect_language(
                    relative_path
                ),
            }
        )

    return sorted(
        discovered_entries,
        key=lambda entry: entry["path"],
    )


def _load_existing_chunks(
    project_id: str,
) -> List[Dict[str, Any]]:
    """
    Load existing chunks for incremental processing.
    """

    chunks_path = get_chunks_path(
        project_id
    )

    if not chunks_path.exists():
        return []

    try:
        existing_chunks = json.loads(
            chunks_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return []

    if not isinstance(
        existing_chunks,
        list,
    ):
        return []

    return [
        chunk
        for chunk in existing_chunks
        if isinstance(chunk, dict)
    ]


def _group_chunks_by_file(
    chunks: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group existing chunks by source file.
    """

    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for chunk in chunks:
        metadata = chunk.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            continue

        file_path = metadata.get(
            "file_path"
        )

        if not isinstance(file_path, str):
            continue

        grouped.setdefault(
            file_path,
            [],
        ).append(chunk)

    return grouped


def _can_reuse_chunks(
    existing_chunks: List[
        Dict[str, Any]
    ],
    file_hash: str,
    config_signature: str,
) -> bool:
    """
    Reuse chunks only when the source and chunking configuration match.
    """

    if not existing_chunks:
        return False

    for chunk in existing_chunks:
        metadata = chunk.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            return False

        if (
            metadata.get(
                "file_hash"
            )
            != file_hash
        ):
            return False

        if (
            metadata.get(
                "chunking_version"
            )
            != CHUNKING_VERSION
        ):
            return False

        if (
            metadata.get(
                "chunking_signature"
            )
            != config_signature
        ):
            return False

    return True


def _deduplicate_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove accidental duplicate records while preserving their order.
    """

    unique_chunks: List[
        Dict[str, Any]
    ] = []

    seen_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = chunk.get(
            "chunk_id"
        )

        if not isinstance(chunk_id, str):
            continue

        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        unique_chunks.append(chunk)

    return unique_chunks


def chunk_file(
    project_id: str,
    file_entry: Dict[str, Any],
    config: ChunkingConfig = (
        DEFAULT_CHUNKING_CONFIG
    ),
) -> List[Dict[str, Any]]:
    """
    Chunk one project file.

    Parsed files use AST-aware chunking. Other supported text files use
    fallback sliding-window chunking.
    """

    config.validate()

    relative_path = file_entry.get(
        "path"
    )

    if (
        not isinstance(relative_path, str)
        or not relative_path.strip()
    ):
        return []

    relative_path = relative_path.strip()

    try:
        source_lines = read_source_lines(
            project_id,
            relative_path,
        )

    except (
        FileNotFoundError,
        OSError,
        UnicodeError,
    ):
        return []

    source_text = "\n".join(
        source_lines
    )

    file_hash = (
        _normalise_text(
            file_entry.get(
                "content_hash"
            )
        )
        or _hash_text(source_text)
    )

    config_signature = (
        _configuration_signature(
            config
        )
    )

    if file_entry.get("status") == "parsed":
        return _chunk_parsed_file(
            project_id=project_id,
            file_entry=file_entry,
            source_lines=source_lines,
            config=config,
            file_hash=file_hash,
            config_signature=(
                config_signature
            ),
        )

    if config.include_unparsed_text_files:
        return _chunk_plain_text_file(
            project_id=project_id,
            file_entry=file_entry,
            source_lines=source_lines,
            config=config,
            file_hash=file_hash,
            config_signature=(
                config_signature
            ),
        )

    return []


def chunk_project(
    project_id: str,
    config: ChunkingConfig = (
        DEFAULT_CHUNKING_CONFIG
    ),
) -> List[Dict[str, Any]]:
    """
    Chunk an entire project using hierarchical AST-aware chunking.

    Unchanged files reuse existing chunks when possible.
    """

    validate_project_id(project_id)
    config.validate()

    ast_data = load_ast(project_id)

    ast_files = [
        file_entry
        for file_entry in ast_data.get(
            "files",
            [],
        )
        if (
            isinstance(file_entry, dict)
            and isinstance(
                file_entry.get("path"),
                str,
            )
        )
    ]

    known_paths = {
        file_entry["path"]
        for file_entry in ast_files
    }

    additional_files = (
        _discover_additional_text_files(
            project_id=project_id,
            known_paths=known_paths,
            config=config,
        )
    )

    all_file_entries = (
        ast_files + additional_files
    )

    existing_chunks = (
        _load_existing_chunks(project_id)
        if config.reuse_unchanged_files
        else []
    )

    existing_by_file = (
        _group_chunks_by_file(
            existing_chunks
        )
    )

    config_signature = (
        _configuration_signature(
            config
        )
    )

    all_chunks: List[
        Dict[str, Any]
    ] = []

    for file_entry in all_file_entries:
        relative_path = file_entry["path"]

        try:
            source_lines = read_source_lines(
                project_id,
                relative_path,
            )

        except (
            FileNotFoundError,
            OSError,
            UnicodeError,
        ):
            continue

        source_text = "\n".join(
            source_lines
        )

        current_file_hash = (
            _normalise_text(
                file_entry.get(
                    "content_hash"
                )
            )
            or _hash_text(source_text)
        )

        previous_file_chunks = (
            existing_by_file.get(
                relative_path,
                [],
            )
        )

        if (
            config.reuse_unchanged_files
            and _can_reuse_chunks(
                previous_file_chunks,
                current_file_hash,
                config_signature,
            )
        ):
            all_chunks.extend(
                previous_file_chunks
            )
            continue

        all_chunks.extend(
            chunk_file(
                project_id=project_id,
                file_entry=file_entry,
                config=config,
            )
        )

    all_chunks = _deduplicate_chunks(
        all_chunks
    )

    chunks_path = get_chunks_path(
        project_id
    )

    chunks_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Atomic write prevents a damaged chunks.json if the process stops during writing.
    
    temporary_path = (
        chunks_path.parent
        / f"{chunks_path.name}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            all_chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        chunks_path
    )

    update_metadata(
        project_id,
        chunk_count=len(all_chunks),
        status="chunked",
    )

    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m "
            "backend.services.chunker "
            "<project_id>"
        )
        sys.exit(1)

    result = chunk_project(
        sys.argv[1]
    )

    print(
        f"Created or reused "
        f"{len(result)} chunks"
    )

    for chunk in result[:5]:
        print(
            json.dumps(
                chunk["metadata"],
                indent=2,
            )
        )