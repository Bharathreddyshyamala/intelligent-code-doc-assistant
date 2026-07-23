import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from .file_scanner import validate_project_id
from .ollama_service import (
    OLLAMA_CHAT_MODEL,
    generate_chat_response,
)
from .retrieval_service import (
    retrieve_code_chunks,
)


load_dotenv()


RAG_MAX_CONTEXT_CHARS = int(
    os.getenv(
        "RAG_MAX_CONTEXT_CHARS",
        "24000",
    )
)

RAG_MAX_CHUNK_CONTEXT_CHARS = int(
    os.getenv(
        "RAG_MAX_CHUNK_CONTEXT_CHARS",
        "5000",
    )
)

RAG_SOURCE_EXCERPT_CHARS = int(
    os.getenv(
        "RAG_SOURCE_EXCERPT_CHARS",
        "1200",
    )
)


SYSTEM_PROMPT = """
You are an intelligent codebase question-answering assistant.

You must answer questions using only the supplied source-code
context.

Rules:

1. Do not invent files, functions, classes, variables, behavior,
   or dependencies.
2. Use the source code and metadata supplied in the context.
3. Mention relevant file names and symbol names.
4. Mention line ranges when they are available.
5. Explain the code clearly and directly.
6. When the context is insufficient, say that the available
   project context is insufficient.
7. Use source references such as [Source 1], [Source 2], and so on.
8. Do not claim that code exists unless it appears in the context.
""".strip()


def get_symbol_name(
    metadata: Dict[str, Any],
) -> str:
    """
    Prefer qualified_name when available.

    Fall back to name because the current chunker may not yet
    store qualified_name.
    """

    return (
        metadata.get("qualified_name")
        or metadata.get("name")
        or "unknown"
    )


def build_rag_context(
    chunks: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert retrieved chunks into LLM context.

    The total context is limited to avoid sending an extremely
    large prompt to the local chat model.
    """

    context_sections: List[str] = []
    used_chunks: List[Dict[str, Any]] = []

    current_context_length = 0

    for chunk in chunks:
        metadata = chunk.get(
            "metadata",
            {},
        )

        document = chunk.get(
            "document",
            "",
        ).strip()

        if not document:
            continue

        document = document[
            :RAG_MAX_CHUNK_CONTEXT_CHARS
        ]

        source_number = len(used_chunks) + 1

        file_path = metadata.get(
            "file_path",
            "unknown",
        )

        symbol_name = get_symbol_name(
            metadata
        )

        source_type = metadata.get(
            "source_type",
            "unknown",
        )

        start_line = metadata.get(
            "start_line",
            "",
        )

        end_line = metadata.get(
            "end_line",
            "",
        )

        section = "\n".join(
            [
                f"[Source {source_number}]",
                f"File: {file_path}",
                f"Symbol: {symbol_name}",
                f"Type: {source_type}",
                f"Lines: {start_line}-{end_line}",
                "",
                document,
            ]
        )

        if (
            current_context_length + len(section)
            > RAG_MAX_CONTEXT_CHARS
        ):
            break

        context_sections.append(
            section
        )

        used_chunks.append(
            chunk
        )

        current_context_length += len(
            section
        )

    return (
        "\n\n".join(context_sections),
        used_chunks,
    )


def build_question_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the final prompt sent to the Ollama chat model.
    """

    return f"""
Answer the following question about the indexed code project.

Question:
{question}

Source-code context:
{context}

Answer requirements:

- Give a direct explanation first.
- Explain how the relevant functions or classes work.
- Mention relevant files and symbols.
- Use [Source N] references that correspond to the supplied context.
- Do not use information outside the supplied context.
""".strip()


def build_source_references(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert retrieved chunks into readable source references
    for the API and Streamlit UI.
    """

    sources: List[Dict[str, Any]] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = chunk.get(
            "metadata",
            {},
        )

        document = chunk.get(
            "document",
            "",
        )

        excerpt = document[
            :RAG_SOURCE_EXCERPT_CHARS
        ]

        if len(document) > RAG_SOURCE_EXCERPT_CHARS:
            excerpt += "\n... [source shortened]"

        sources.append(
            {
                "source_number": index,
                "chunk_id": chunk.get(
                    "chunk_id",
                    "",
                ),
                "file_path": metadata.get(
                    "file_path",
                    "",
                ),
                "symbol": get_symbol_name(
                    metadata
                ),
                "source_type": metadata.get(
                    "source_type",
                    "",
                ),
                "start_line": int(
                    metadata.get(
                        "start_line",
                        0,
                    )
                ),
                "end_line": int(
                    metadata.get(
                        "end_line",
                        0,
                    )
                ),
                "distance": round(
                    float(
                        chunk.get(
                            "distance",
                            0.0,
                        )
                    ),
                    6,
                ),
                "excerpt": excerpt,
            }
        )

    return sources


def answer_code_question(
    project_id: str,
    question: str,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Run the complete Ask Code RAG workflow.

    1. Validate the project and question.
    2. Retrieve matching code chunks.
    3. Build the LLM context.
    4. Generate an answer using Ollama.
    5. Return the answer with source references.
    """

    validate_project_id(
        project_id
    )

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    retrieved_chunks = retrieve_code_chunks(
        project_id=project_id,
        query=cleaned_question,
        top_k=top_k,
    )

    if not retrieved_chunks:
        raise ValueError(
            "No indexed code chunks were found for this "
            "project. Index the project before asking questions."
        )

    context, used_chunks = build_rag_context(
        retrieved_chunks
    )

    if not context:
        raise ValueError(
            "The retrieved code chunks did not contain "
            "usable source-code context."
        )

    prompt = build_question_prompt(
        question=cleaned_question,
        context=context,
    )

    answer = generate_chat_response(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
    )

    sources = build_source_references(
        used_chunks
    )

    return {
        "project_id": project_id,
        "question": cleaned_question,
        "answer": answer,
        "chat_model": OLLAMA_CHAT_MODEL,
        "retrieved_chunk_count": len(
            retrieved_chunks
        ),
        "used_chunk_count": len(
            used_chunks
        ),
        "sources": sources,
    }