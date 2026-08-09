import html
import json
import re
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.xpreformatted import XPreformatted

from services.file_scanner import (
    get_ast_path,
    get_source_directory,
    validate_project_id,
)
from services.ollama_service import generate_chat_response


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
GENERATED_DOCS_DIRECTORY = BACKEND_DIRECTORY / "generated_docs"


DOCUMENTATION_SYSTEM_PROMPT = """
You are a senior software engineer and technical documentation writer.

Generate accurate, beginner-friendly Markdown documentation from the supplied
source code and AST metadata.

Important rules:

1. Use only information present in the supplied code and metadata.
2. Do not invent classes, functions, parameters, return values, or behaviour.
3. Clearly document classes, methods, functions, imports, and dependencies.
4. Explain parameters, return values, exceptions, and execution flow.
5. Preserve useful information from existing docstrings.
6. Mention when a function or class has no docstring.
7. Return clean Markdown.
8. Do not wrap the entire response in one outer Markdown code block.
9. Prefer headings, short paragraphs, and bullet lists.
10. Avoid complex Markdown tables when a bullet list communicates the same content.
""".strip()


PROJECT_SUMMARY_SYSTEM_PROMPT = """
You are a senior software architect and technical documentation writer.

Generate a project-level technical summary using only the supplied project
structure and generated file summaries.

Important rules:

1. Explain the purpose of the project.
2. Explain the main files and components.
3. Explain how the files are connected.
4. Explain the overall execution flow.
5. Mention important internal and external dependencies.
6. Do not invent features, routes, commands, or components.
7. Return clean Markdown.
8. Begin with a second-level heading, not a first-level project title.
9. Prefer headings, short paragraphs, and bullet lists.
""".strip()


class DocumentationGenerationError(RuntimeError):
    """Raised when project documentation cannot be generated."""


def get_generated_docs_directory(project_id: str) -> Path:
    """Return the generated-documentation directory for a project."""
    validate_project_id(project_id)
    return GENERATED_DOCS_DIRECTORY / project_id


def get_generated_documentation_path(project_id: str) -> Path:
    """
    Return the final generated PDF path.

    The function name is intentionally preserved so existing route imports
    do not break.
    """
    return get_generated_docs_directory(project_id) / "PROJECT_DOCUMENTATION.pdf"


def get_generated_pdf_path(project_id: str) -> Path:
    """Explicit alias for the final generated PDF path."""
    return get_generated_documentation_path(project_id)


def _load_ast_data(project_id: str) -> Dict[str, Any]:
    """Read and validate the project's ast.json file."""
    ast_path = get_ast_path(project_id)

    if not ast_path.exists():
        raise DocumentationGenerationError(
            "ast.json was not found. Parse the project before generating documentation."
        )

    try:
        with ast_path.open("r", encoding="utf-8") as file:
            ast_data = json.load(file)
    except json.JSONDecodeError as error:
        raise DocumentationGenerationError(
            "ast.json contains invalid JSON: {}".format(error)
        ) from error
    except OSError as error:
        raise DocumentationGenerationError(
            "Could not read ast.json: {}".format(error)
        ) from error

    if not isinstance(ast_data, dict):
        raise DocumentationGenerationError(
            "ast.json must contain a JSON object."
        )

    return ast_data


def _truncate_text(text: str, maximum_characters: int) -> str:
    """Limit text before sending it to the local LLM."""
    if len(text) <= maximum_characters:
        return text

    return (
        text[:maximum_characters]
        + "\n\n[Content truncated because the input is large.]"
    )


def _remove_large_ast_fields(value: Any) -> Any:
    """Remove repeated code and internal identifiers from prompt metadata."""
    excluded_keys = {
        "id",
        "content_hash",
        "code",
    }

    if isinstance(value, dict):
        return {
            key: _remove_large_ast_fields(item)
            for key, item in value.items()
            if key not in excluded_keys
        }

    if isinstance(value, list):
        return [_remove_large_ast_fields(item) for item in value]

    return value


def _safe_source_path(source_directory: Path, relative_file_path: str) -> Path:
    """Resolve a project file while preventing path traversal."""
    source_root = source_directory.resolve()
    requested_path = (source_root / relative_file_path).resolve()

    if requested_path != source_root and source_root not in requested_path.parents:
        raise DocumentationGenerationError(
            "Unsafe source file path: {}".format(relative_file_path)
        )

    return requested_path


def _read_source_code(source_directory: Path, relative_file_path: str) -> str:
    """Safely read one source file."""
    requested_path = _safe_source_path(source_directory, relative_file_path)

    if not requested_path.exists() or not requested_path.is_file():
        return ""

    try:
        return requested_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""


def _extract_code_from_ast(file_data: Dict[str, Any]) -> str:
    """Use code stored in AST records when the source file is unavailable."""
    code_sections = []  # type: List[str]

    for class_data in file_data.get("classes", []):
        if not isinstance(class_data, dict):
            continue

        class_code = class_data.get("code")
        if isinstance(class_code, str) and class_code.strip():
            code_sections.append(class_code)

    for function_data in file_data.get("functions", []):
        if not isinstance(function_data, dict):
            continue

        function_code = function_data.get("code")
        if isinstance(function_code, str) and function_code.strip():
            code_sections.append(function_code)

    return "\n\n".join(code_sections)


def _clean_markdown_response(response: Any) -> str:
    """Normalise the Ollama response and remove one outer Markdown fence."""
    if response is None:
        return ""

    cleaned_response = str(response).strip()

    if cleaned_response.startswith("```markdown") and cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[len("```markdown"):-3].strip()
    elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[3:-3].strip()

    return cleaned_response


def _language_for_markdown_fence(file_path: str) -> str:
    """Return a useful Markdown code-fence language."""
    suffix = Path(file_path).suffix.lower()

    languages = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".html": "html",
        ".css": "css",
        ".sh": "bash",
    }

    return languages.get(suffix, "text")


def _build_file_documentation_prompt(
    file_data: Dict[str, Any],
    source_code: str,
) -> str:
    """Build the Ollama prompt for one source file."""
    file_path = str(file_data.get("path", "unknown_file.py"))

    cleaned_ast_data = _remove_large_ast_fields(file_data)
    ast_metadata = json.dumps(
        cleaned_ast_data,
        indent=2,
        ensure_ascii=False,
    )
    ast_metadata = _truncate_text(ast_metadata, maximum_characters=12000)
    source_code = _truncate_text(source_code, maximum_characters=20000)
    language = _language_for_markdown_fence(file_path)

    sections = [
        "Generate detailed Markdown documentation for the following source file.",
        "File path:\n\n{}".format(file_path),
        "AST metadata:\n\n```json\n{}\n```".format(ast_metadata),
        "Source code:\n\n```{}\n{}\n```".format(language, source_code),
        (
            "Generate the documentation using this structure:\n\n"
            "# {}\n\n"
            "## Purpose\n\n"
            "Explain the responsibility of this file.\n\n"
            "## Imports and Dependencies\n\n"
            "Explain the important imports and why they are used.\n\n"
            "## Module Variables\n\n"
            "Document important module-level variables and constants.\n\n"
            "## Classes\n\n"
            "For every class, document its purpose, parent classes, constructor, "
            "attributes, methods, and important behaviour. Omit this section when "
            "there are no classes.\n\n"
            "## Functions\n\n"
            "For every function, document its purpose, parameters, return value, "
            "exceptions, processing steps, and called services. Omit this section "
            "when there are no functions.\n\n"
            "## Execution Flow\n\n"
            "Explain how this file processes requests or data.\n\n"
            "## Usage\n\n"
            "Include usage only when it can be accurately inferred.\n\n"
            "## Documentation Notes\n\n"
            "Mention missing docstrings, unclear behaviour, limitations, and useful "
            "improvements."
        ).format(file_path),
        (
            "Important rules:\n\n"
            "- Use only the supplied AST metadata and source code.\n"
            "- Do not invent functions, classes, parameters, routes, or behaviour.\n"
            "- Return Markdown only.\n"
            "- Do not wrap the complete response inside another Markdown code block."
        ),
    ]

    return "\n\n".join(sections).strip()


def _create_project_structure_context(ast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact project structure for the project summary prompt."""
    file_summaries = []  # type: List[Dict[str, Any]]

    for file_data in ast_data.get("files", []):
        if not isinstance(file_data, dict):
            continue

        if file_data.get("status") != "parsed":
            continue

        classes = []
        for class_data in file_data.get("classes", []):
            if isinstance(class_data, dict):
                classes.append(
                    {
                        "name": class_data.get("name"),
                        "qualified_name": class_data.get("qualified_name"),
                        "bases": class_data.get("bases", []),
                        "methods": [
                            method.get("name")
                            for method in class_data.get("methods", [])
                            if isinstance(method, dict)
                        ],
                    }
                )

        functions = [
            {
                "name": function_data.get("name"),
                "qualified_name": function_data.get("qualified_name"),
                "is_async": function_data.get("is_async", False),
            }
            for function_data in file_data.get("functions", [])
            if isinstance(function_data, dict)
        ]

        file_summaries.append(
            {
                "path": file_data.get("path"),
                "module_name": file_data.get("module_name"),
                "module_docstring": file_data.get("module_docstring"),
                "imports": file_data.get("imports", []),
                "module_variables": file_data.get("module_variables", []),
                "classes": classes,
                "functions": functions,
            }
        )

    return {
        "language": ast_data.get("language", "python"),
        "file_count": ast_data.get("file_count", len(file_summaries)),
        "files": file_summaries,
        "dependencies": ast_data.get("dependencies", []),
        "external_dependencies": ast_data.get("external_dependencies", []),
        "routes": ast_data.get("routes", []),
        "call_graph": ast_data.get("call_graph", []),
        "inheritance_graph": ast_data.get("inheritance_graph", []),
        "statistics": ast_data.get("statistics", {}),
    }


def _build_project_summary_prompt(
    project_title: str,
    ast_data: Dict[str, Any],
    file_summaries: List[Dict[str, str]],
) -> str:
    """Build the project-level documentation prompt."""
    project_structure = _create_project_structure_context(ast_data)
    project_structure_text = json.dumps(
        project_structure,
        indent=2,
        ensure_ascii=False,
    )
    project_structure_text = _truncate_text(
        project_structure_text,
        maximum_characters=30000,
    )

    summary_sections = []
    for item in file_summaries:
        summary_sections.append(
            "### {}\n\n{}".format(
                item["source_file"],
                _truncate_text(item["markdown"], maximum_characters=2000),
            )
        )

    generated_file_summary_text = _truncate_text(
        "\n\n".join(summary_sections),
        maximum_characters=30000,
    )

    sections = [
        "Generate project-level Markdown documentation for: {}".format(project_title),
        "Project structure:\n\n```json\n{}\n```".format(project_structure_text),
        "Generated file-documentation summaries:\n\n{}".format(
            generated_file_summary_text
        ),
        (
            "Use this structure:\n\n"
            "## Project Summary\n\n"
            "## Main Features\n\n"
            "## Architecture\n\n"
            "## Folder and File Responsibilities\n\n"
            "## Processing Flow\n\n"
            "## Internal Dependencies\n\n"
            "## External Dependencies\n\n"
            "## API Endpoints\n\n"
            "## Important Classes and Functions\n\n"
            "## Error Handling\n\n"
            "## Limitations and Documentation Notes"
        ),
        (
            "Important rules:\n\n"
            "- Use only the supplied project information.\n"
            "- Do not invent files, routes, commands, or functionality.\n"
            "- Return Markdown only.\n"
            "- Do not add a top-level project heading because the application adds it."
        ),
    ]

    return "\n\n".join(sections).strip()


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _register_document_fonts() -> Dict[str, str]:
    """
    Register common Unicode fonts when they are available.

    Built-in ReportLab fonts remain the fallback, so the feature still works
    on systems where none of these optional font files exist.
    """
    regular_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    mono_candidates = [
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/Library/Fonts/Courier New.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]

    fonts = {
        "regular": "Helvetica",
        "bold": "Helvetica-Bold",
        "mono": "Courier",
    }

    regular_path = next((path for path in regular_candidates if path.exists()), None)
    bold_path = next((path for path in bold_candidates if path.exists()), None)
    mono_path = next((path for path in mono_candidates if path.exists()), None)

    try:
        if regular_path and bold_path:
            pdfmetrics.registerFont(TTFont("DocumentationSans", str(regular_path)))
            pdfmetrics.registerFont(TTFont("DocumentationSansBold", str(bold_path)))
            pdfmetrics.registerFontFamily(
                "DocumentationSans",
                normal="DocumentationSans",
                bold="DocumentationSansBold",
                italic="DocumentationSans",
                boldItalic="DocumentationSansBold",
            )
            fonts["regular"] = "DocumentationSans"
            fonts["bold"] = "DocumentationSansBold"

        if mono_path:
            pdfmetrics.registerFont(TTFont("DocumentationMono", str(mono_path)))
            fonts["mono"] = "DocumentationMono"
    except Exception:
        # The built-in fonts remain available when optional registration fails.
        pass

    return fonts


def _normalise_pdf_text(value: str) -> str:
    """Normalise common Unicode punctuation for built-in PDF fonts."""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "\u2192": "->",
        "\u21d2": "=>",
        "\u2022": "*",
    }

    text = value
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    return "".join(
        character
        for character in text
        if character == "\n"
        or character == "\t"
        or ord(character) >= 32
    )


def _inline_markdown_to_reportlab(text: str, mono_font: str) -> str:
    """Convert a small, safe subset of inline Markdown to ReportLab markup."""
    safe_text = html.escape(_normalise_pdf_text(text))

    safe_text = re.sub(
        r"`([^`]+)`",
        lambda match: '<font name="{}">{}</font>'.format(
            mono_font,
            match.group(1),
        ),
        safe_text,
    )
    safe_text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", safe_text)
    safe_text = re.sub(r"__([^_]+)__", r"<b>\1</b>", safe_text)
    safe_text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe_text)

    return safe_text


def _wrap_code_for_pdf(code: str, width: int = 105) -> str:
    """Wrap exceptionally long source-code lines to avoid clipped PDF text."""
    wrapped_lines = []

    for line in _normalise_pdf_text(code).expandtabs(4).splitlines():
        if len(line) <= width:
            wrapped_lines.append(line)
            continue

        indentation_length = len(line) - len(line.lstrip(" "))
        indentation = " " * indentation_length
        continuation_indent = indentation + "    "

        segments = textwrap.wrap(
            line,
            width=width,
            subsequent_indent=continuation_indent,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(segments or [line])

    return "\n".join(wrapped_lines)


def _build_pdf_styles(fonts: Dict[str, str]) -> Dict[str, ParagraphStyle]:
    """Create ReportLab styles for the generated documentation."""
    sample_styles = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "DocumentationTitle",
            parent=sample_styles["Title"],
            fontName=fonts["bold"],
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            spaceAfter=14,
            textColor=colors.HexColor("#1f2937"),
        ),
        "h1": ParagraphStyle(
            "DocumentationH1",
            parent=sample_styles["Heading1"],
            fontName=fonts["bold"],
            fontSize=17,
            leading=21,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#111827"),
        ),
        "h2": ParagraphStyle(
            "DocumentationH2",
            parent=sample_styles["Heading2"],
            fontName=fonts["bold"],
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#1f2937"),
        ),
        "h3": ParagraphStyle(
            "DocumentationH3",
            parent=sample_styles["Heading3"],
            fontName=fonts["bold"],
            fontSize=12,
            leading=16,
            spaceBefore=10,
            spaceAfter=5,
            textColor=colors.HexColor("#374151"),
        ),
        "body": ParagraphStyle(
            "DocumentationBody",
            parent=sample_styles["BodyText"],
            fontName=fonts["regular"],
            fontSize=9.5,
            leading=14,
            spaceAfter=7,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
        ),
        "small": ParagraphStyle(
            "DocumentationSmall",
            parent=sample_styles["BodyText"],
            fontName=fonts["regular"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#4b5563"),
        ),
        "code": ParagraphStyle(
            "DocumentationCode",
            parent=sample_styles["Code"],
            fontName=fonts["mono"],
            fontSize=7.3,
            leading=9.4,
            leftIndent=5,
            rightIndent=5,
            spaceBefore=5,
            spaceAfter=8,
            backColor=colors.HexColor("#f3f4f6"),
            borderColor=colors.HexColor("#d1d5db"),
            borderWidth=0.5,
            borderPadding=6,
        ),
        "quote": ParagraphStyle(
            "DocumentationQuote",
            parent=sample_styles["BodyText"],
            fontName=fonts["regular"],
            fontSize=9,
            leading=13,
            leftIndent=12,
            borderColor=colors.HexColor("#9ca3af"),
            borderWidth=1,
            borderPadding=6,
            textColor=colors.HexColor("#374151"),
            spaceAfter=7,
        ),
    }

    return styles


def _looks_like_table_separator(line: str) -> bool:
    """Return True for a Markdown table separator row."""
    stripped = line.strip().strip("|")
    if not stripped:
        return False

    cells = [cell.strip() for cell in stripped.split("|")]
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) is not None
        for cell in cells
    )


def _parse_markdown_table(
    lines: Sequence[str],
    start_index: int,
    styles: Dict[str, ParagraphStyle],
    fonts: Dict[str, str],
) -> Tuple[Optional[Table], int]:
    """Parse a basic pipe-separated Markdown table."""
    if start_index + 1 >= len(lines):
        return None, start_index

    header_line = lines[start_index]
    separator_line = lines[start_index + 1]

    if "|" not in header_line or not _looks_like_table_separator(separator_line):
        return None, start_index

    table_lines = [header_line]
    cursor = start_index + 2

    while cursor < len(lines):
        candidate = lines[cursor]
        if not candidate.strip() or "|" not in candidate:
            break

        table_lines.append(candidate)
        cursor += 1

    rows = []
    for line_number, line in enumerate(table_lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        row_style = styles["small"]
        row = [
            Paragraph(
                _inline_markdown_to_reportlab(cell, fonts["mono"]),
                row_style,
            )
            for cell in cells
        ]
        rows.append(row)

    if not rows:
        return None, start_index

    column_count = max(len(row) for row in rows)
    for row in rows:
        while len(row) < column_count:
            row.append(Paragraph("", styles["small"]))

    available_width = A4[0] - 36 * mm
    column_widths = [available_width / column_count] * column_count

    table = Table(
        rows,
        colWidths=column_widths,
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), fonts["bold"]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return table, cursor - 1


def _markdown_to_flowables(
    markdown_text: str,
    styles: Dict[str, ParagraphStyle],
    fonts: Dict[str, str],
) -> List[Any]:
    """Convert generated Markdown into ReportLab flowables."""
    lines = _normalise_pdf_text(markdown_text).splitlines()
    flowables = []  # type: List[Any]
    cursor = 0

    while cursor < len(lines):
        line = lines[cursor]
        stripped = line.strip()

        if not stripped:
            flowables.append(Spacer(1, 3))
            cursor += 1
            continue

        if stripped.startswith("```"):
            code_lines = []
            cursor += 1

            while cursor < len(lines) and not lines[cursor].strip().startswith("```"):
                code_lines.append(lines[cursor])
                cursor += 1

            if cursor < len(lines):
                cursor += 1

            flowables.append(
                XPreformatted(
                    html.escape(_wrap_code_for_pdf("\n".join(code_lines))),
                    styles["code"],
                )
            )
            continue

        table, table_end = _parse_markdown_table(
            lines=lines,
            start_index=cursor,
            styles=styles,
            fonts=fonts,
        )
        if table is not None:
            flowables.extend([table, Spacer(1, 8)])
            cursor = table_end + 1
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            style_name = {
                1: "h1",
                2: "h2",
                3: "h3",
                4: "h3",
            }[level]

            flowables.append(
                Paragraph(
                    _inline_markdown_to_reportlab(heading_text, fonts["mono"]),
                    styles[style_name],
                )
            )
            cursor += 1
            continue

        if re.fullmatch(r"[-*_]{3,}", stripped):
            flowables.extend(
                [
                    Spacer(1, 5),
                    HRFlowable(
                        width="100%",
                        thickness=0.6,
                        color=colors.HexColor("#9ca3af"),
                    ),
                    Spacer(1, 7),
                ]
            )
            cursor += 1
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while cursor < len(lines) and lines[cursor].strip().startswith(">"):
                quote_lines.append(lines[cursor].strip().lstrip(">").strip())
                cursor += 1

            flowables.append(
                Paragraph(
                    _inline_markdown_to_reportlab(
                        " ".join(quote_lines),
                        fonts["mono"],
                    ),
                    styles["quote"],
                )
            )
            continue

        if re.match(r"^[-*+]\s+", stripped):
            items = []

            while cursor < len(lines):
                bullet_match = re.match(r"^\s*[-*+]\s+(.+)$", lines[cursor])
                if not bullet_match:
                    break

                items.append(
                    ListItem(
                        Paragraph(
                            _inline_markdown_to_reportlab(
                                bullet_match.group(1),
                                fonts["mono"],
                            ),
                            styles["body"],
                        ),
                        leftIndent=12,
                    )
                )
                cursor += 1

            flowables.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                    bulletFontName=fonts["regular"],
                    bulletFontSize=7,
                    spaceAfter=6,
                )
            )
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []

            while cursor < len(lines):
                number_match = re.match(r"^\s*\d+\.\s+(.+)$", lines[cursor])
                if not number_match:
                    break

                items.append(
                    ListItem(
                        Paragraph(
                            _inline_markdown_to_reportlab(
                                number_match.group(1),
                                fonts["mono"],
                            ),
                            styles["body"],
                        ),
                        leftIndent=12,
                    )
                )
                cursor += 1

            flowables.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    leftIndent=20,
                    bulletFontName=fonts["regular"],
                    bulletFontSize=8,
                    spaceAfter=6,
                )
            )
            continue

        paragraph_lines = [stripped]
        cursor += 1

        while cursor < len(lines):
            candidate = lines[cursor]
            candidate_stripped = candidate.strip()

            if not candidate_stripped:
                break

            if (
                candidate_stripped.startswith("```")
                or re.match(r"^(#{1,4})\s+", candidate_stripped)
                or re.fullmatch(r"[-*_]{3,}", candidate_stripped)
                or re.match(r"^[-*+]\s+", candidate_stripped)
                or re.match(r"^\d+\.\s+", candidate_stripped)
                or candidate_stripped.startswith(">")
            ):
                break

            if (
                "|" in candidate_stripped
                and cursor + 1 < len(lines)
                and _looks_like_table_separator(lines[cursor + 1])
            ):
                break

            paragraph_lines.append(candidate_stripped)
            cursor += 1

        flowables.append(
            Paragraph(
                _inline_markdown_to_reportlab(
                    " ".join(paragraph_lines),
                    fonts["mono"],
                ),
                styles["body"],
            )
        )

    return flowables


def _draw_page_header_footer(canvas: Any, document: Any, title: str, fonts: Dict[str, str]) -> None:
    """Draw a small header and page number on each PDF page."""
    canvas.saveState()
    page_width, page_height = A4

    canvas.setFont(fonts["regular"], 7.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(
        18 * mm,
        page_height - 11 * mm,
        _normalise_pdf_text(title)[:80],
    )
    canvas.drawRightString(
        page_width - 18 * mm,
        10 * mm,
        "Page {}".format(document.page),
    )

    canvas.setStrokeColor(colors.HexColor("#d1d5db"))
    canvas.setLineWidth(0.4)
    canvas.line(
        18 * mm,
        page_height - 13 * mm,
        page_width - 18 * mm,
        page_height - 13 * mm,
    )
    canvas.restoreState()


def _build_pdf_document(
    markdown_text: str,
    output_path: Path,
    project_title: str,
    generated_at: str,
) -> None:
    """Render generated Markdown into a paginated PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    fonts = _register_document_fonts()
    styles = _build_pdf_styles(fonts)

    document = SimpleDocTemplate(
        str(temporary_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title=project_title,
        author="Intelligent Code Documentation Assistant",
        subject="Generated source-code documentation",
    )

    story = [
        Spacer(1, 8 * mm),
        Paragraph(
            _inline_markdown_to_reportlab(project_title, fonts["mono"]),
            styles["title"],
        ),
        Paragraph(
            "Generated locally using Ollama on {}".format(
                html.escape(generated_at)
            ),
            styles["small"],
        ),
        Spacer(1, 8 * mm),
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.HexColor("#9ca3af"),
        ),
        Spacer(1, 8 * mm),
    ]

    story.extend(
        _markdown_to_flowables(
            markdown_text=markdown_text,
            styles=styles,
            fonts=fonts,
        )
    )

    document.build(
        story,
        onFirstPage=lambda canvas, doc: _draw_page_header_footer(
            canvas,
            doc,
            project_title,
            fonts,
        ),
        onLaterPages=lambda canvas, doc: _draw_page_header_footer(
            canvas,
            doc,
            project_title,
            fonts,
        ),
    )

    temporary_path.replace(output_path)


def _file_priority(file_data: Dict[str, Any]) -> Tuple[int, str]:
    """Prioritise central application files before tests and examples."""
    file_path = str(file_data.get("path", "")).lower()
    priority = 50

    if file_path.endswith(("main.py", "app.py", "server.py", "manage.py")):
        priority = 0
    elif any(part in file_path for part in ("routes/", "controllers/", "api/")):
        priority = 10
    elif any(part in file_path for part in ("services/", "core/", "agents/")):
        priority = 20
    elif any(
        part in file_path
        for part in ("models/", "schemas/", "repositories/")
    ):
        priority = 30
    elif any(
        part in file_path
        for part in ("tests/", "test/", "migrations/", "examples/", "scripts/")
    ):
        priority = 90

    return priority, file_path


def generate_project_documentation(
    project_id: str,
    project_title: Optional[str] = None,
    max_files: int = 30,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Generate project documentation and save the final output as a PDF.

    Ollama still returns Markdown because it is a reliable structured text
    format. The Markdown remains in memory and is rendered to PDF using
    ReportLab. No final Markdown file is written.
    """
    validate_project_id(project_id)

    if max_files < 1:
        raise ValueError("max_files must be greater than zero.")

    ast_data = _load_ast_data(project_id)
    source_directory = get_source_directory(project_id)

    if not source_directory.exists() or not source_directory.is_dir():
        raise DocumentationGenerationError(
            "The project source directory was not found."
        )

    parsed_files = [
        file_data
        for file_data in ast_data.get("files", [])
        if isinstance(file_data, dict) and file_data.get("status") == "parsed"
    ]

    if not parsed_files:
        raise DocumentationGenerationError(
            "No successfully parsed files were found inside ast.json."
        )

    selected_files = sorted(parsed_files, key=_file_priority)[:max_files]
    output_directory = get_generated_docs_directory(project_id)
    main_documentation_path = get_generated_documentation_path(project_id)

    if output_directory.exists():
        if overwrite:
            shutil.rmtree(str(output_directory))
        elif main_documentation_path.exists():
            raise DocumentationGenerationError(
                "Documentation already exists. Set overwrite to true to regenerate it."
            )

    output_directory.mkdir(parents=True, exist_ok=True)

    final_project_title = (
        project_title.strip()
        if project_title and project_title.strip()
        else "Project {}".format(project_id[:8])
    )

    generated_file_documents = []  # type: List[Dict[str, str]]
    skipped_files = []  # type: List[str]

    for file_number, file_data in enumerate(selected_files, start=1):
        relative_file_path = str(
            file_data.get("path", "unknown_file_{}.py".format(file_number))
        )

        source_code = _read_source_code(
            source_directory=source_directory,
            relative_file_path=relative_file_path,
        )

        if not source_code.strip():
            source_code = _extract_code_from_ast(file_data)

        if not source_code.strip():
            skipped_files.append(relative_file_path)
            continue

        prompt = _build_file_documentation_prompt(
            file_data=file_data,
            source_code=source_code,
        )

        try:
            generated_markdown = generate_chat_response(
                prompt=prompt,
                system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as error:
            raise DocumentationGenerationError(
                "Ollama failed while documenting {}: {}".format(
                    relative_file_path,
                    error,
                )
            ) from error

        generated_markdown = _clean_markdown_response(generated_markdown)

        if not generated_markdown:
            skipped_files.append(relative_file_path)
            continue

        generated_file_documents.append(
            {
                "source_file": relative_file_path,
                "markdown": generated_markdown,
            }
        )

    if not generated_file_documents:
        raise DocumentationGenerationError(
            "No source-file documentation could be generated."
        )

    project_summary_prompt = _build_project_summary_prompt(
        project_title=final_project_title,
        ast_data=ast_data,
        file_summaries=generated_file_documents,
    )

    try:
        project_summary = generate_chat_response(
            prompt=project_summary_prompt,
            system_prompt=PROJECT_SUMMARY_SYSTEM_PROMPT,
            temperature=0.1,
        )
    except Exception as error:
        raise DocumentationGenerationError(
            "Ollama failed while generating the project summary: {}".format(error)
        ) from error

    project_summary = _clean_markdown_response(project_summary)

    if not project_summary:
        raise DocumentationGenerationError(
            "Ollama returned an empty project summary."
        )

    generated_at = datetime.now(timezone.utc).isoformat()

    complete_sections = [
        project_summary,
        "---",
        "# Detailed File Documentation",
    ]

    for item in generated_file_documents:
        complete_sections.extend(
            [
                item["markdown"],
                "---",
            ]
        )

    complete_documentation = "\n\n".join(complete_sections).strip() + "\n"

    try:
        _build_pdf_document(
            markdown_text=complete_documentation,
            output_path=main_documentation_path,
            project_title=final_project_title,
            generated_at=generated_at,
        )
    except Exception as error:
        raise DocumentationGenerationError(
            "PDF creation failed: {}".format(error)
        ) from error

    manifest = {
        "project_id": project_id,
        "project_title": final_project_title,
        "generated_at": generated_at,
        "status": "completed",
        "output_format": "pdf",
        "documented_file_count": len(generated_file_documents),
        "skipped_file_count": len(skipped_files),
        "documented_files": [
            item["source_file"]
            for item in generated_file_documents
        ],
        "skipped_files": skipped_files,
        "pdf_file": main_documentation_path.name,
    }

    _write_json_atomic(output_directory / "manifest.json", manifest)

    return {
        "project_id": project_id,
        "project_title": final_project_title,
        "status": "completed",
        "output_format": "pdf",
        "documented_file_count": len(generated_file_documents),
        "skipped_file_count": len(skipped_files),
        "output_path": str(main_documentation_path),
        "preview_url": "/generated-docs/{}".format(project_id),
        "download_url": "/generated-docs/{}/download".format(project_id),
    }