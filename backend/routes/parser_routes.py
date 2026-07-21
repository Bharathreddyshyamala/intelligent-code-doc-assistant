import json
from typing import Any, Dict

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import BaseModel, Field

from services.ast_parser import ast_parser
from services.file_scanner import (
    get_ast_path,
    get_source_directory,
    read_metadata,
    update_metadata,
    validate_project_id,
)


router = APIRouter(
    tags=["Parser"],
)


class ParseCodeRequest(BaseModel):
    project_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="Project ID returned by an ingestion API",
    )


class ParseCodeResponse(BaseModel):
    project_id: str
    file_count: int
    status: str


def validate_parser_result(
    result: Any,
) -> Dict[str, Any]:
    """
    Validate the response returned by Member 1's parser.
    """

    if not isinstance(result, dict):
        raise RuntimeError(
            "ast_parser.parse_project() must return a dictionary"
        )

    required_fields = {
        "project_id",
        "file_count",
        "error_count",
        "files",
    }

    missing_fields = required_fields.difference(
        result.keys()
    )

    if missing_fields:
        raise RuntimeError(
            "Parser result is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    if not isinstance(result["file_count"], int):
        raise RuntimeError(
            "Parser result file_count must be an integer"
        )

    if not isinstance(result["error_count"], int):
        raise RuntimeError(
            "Parser result error_count must be an integer"
        )

    if not isinstance(result["files"], list):
        raise RuntimeError(
            "Parser result files must be a list"
        )

    return result


@router.post(
    "/parse-code",
    response_model=ParseCodeResponse,
    status_code=status.HTTP_200_OK,
)
def parse_code(
    request: ParseCodeRequest,
) -> ParseCodeResponse:
    """
    Parse a previously ingested project.

    Flow:
    1. Validate project_id
    2. Verify project metadata exists
    3. Find the source directory
    4. Call Member 1's parse_project()
    5. Save or verify ast.json
    6. Update metadata.json
    7. Return project_id, file_count and status
    """

    try:
        validate_project_id(
            request.project_id
        )

        metadata = read_metadata(
            request.project_id
        )

        source_directory = get_source_directory(
            request.project_id
        )

        if not source_directory.exists():
            raise FileNotFoundError(
                "The project source directory was not found"
            )

        ast_output_path = get_ast_path(
            request.project_id
        )

        update_metadata(
            request.project_id,
            status="parsing",
        )

        # Direct dependency on Member 1.
        parser_result = ast_parser.parse_project(
    request.project_id
)

        parser_result = validate_parser_result(
            parser_result
        )

        if (
            parser_result["project_id"]
            != request.project_id
        ):
            raise RuntimeError(
                "Parser returned a different project_id"
            )

        # This fallback writes ast.json when Member 1 returns
        # the parsed data but does not write the output file.
        if not ast_output_path.exists():
            ast_output_path.write_text(
                json.dumps(
                    parser_result,
                    indent=2,
                ),
                encoding="utf-8",
            )

        parsed_file_count = parser_result[
            "file_count"
        ]

        parsing_error_count = parser_result[
            "error_count"
        ]

        final_status = (
            "parsed"
            if parsed_file_count > 0
            else "parse_failed"
        )

        update_metadata(
            request.project_id,
            status=final_status,
            parsed_file_count=parsed_file_count,
            parsing_error_count=parsing_error_count,
            ast_output_path=str(ast_output_path),
            original_file_count=metadata.get(
                "file_count",
                0,
            ),
        )

        if parsed_file_count == 0:
            raise ValueError(
                "No files were successfully parsed"
            )

        return ParseCodeResponse(
            project_id=request.project_id,
            file_count=parsed_file_count,
            status="parsed",
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        try:
            update_metadata(
                request.project_id,
                status="parse_failed",
                parsing_error=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        try:
            update_metadata(
                request.project_id,
                status="parse_failed",
                parsing_error=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Parser integration error: "
                f"{exc}"
            ),
        ) from exc

    except Exception as exc:
        try:
            update_metadata(
                request.project_id,
                status="parse_failed",
                parsing_error=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code parsing failed: {exc}",
        ) from exc