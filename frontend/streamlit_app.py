import os
from typing import Any, Dict
from ask_code_ui import (
    render_ask_code_section,
)

import requests
import streamlit as st


# FastAPI backend URL
FASTAPI_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="Intelligent Code Documentation Assistant",
    page_icon="📘",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "project_id" not in st.session_state:
    st.session_state["project_id"] = None

if "project_status" not in st.session_state:
    st.session_state["project_status"] = None

if "file_count" not in st.session_state:
    st.session_state["file_count"] = 0

if "source_type" not in st.session_state:
    st.session_state["source_type"] = None

if "parsed_file_count" not in st.session_state:
    st.session_state["parsed_file_count"] = 0

if "parsing_error_count" not in st.session_state:
    st.session_state["parsing_error_count"] = 0

if "parse_completed" not in st.session_state:
    st.session_state["parse_completed"] = False

if "indexing_status" not in st.session_state:
    st.session_state["indexing_status"] = "not_started"

if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0

if "embedding_count" not in st.session_state:
    st.session_state["embedding_count"] = 0

if "indexed_count" not in st.session_state:
    st.session_state["indexed_count"] = 0

if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = None

if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = None


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def get_response_json(
    response: requests.Response,
) -> Dict[str, Any]:
    """
    Safely convert the FastAPI response into JSON.
    """

    try:
        return response.json()
    except ValueError:
        return {
            "detail": "The backend returned an invalid response."
        }


def reset_processing_state() -> None:
    """
    Reset parsing and indexing information for a newly ingested project.
    """

    st.session_state["parsed_file_count"] = 0
    st.session_state["parsing_error_count"] = 0
    st.session_state["parse_completed"] = False
    st.session_state["indexing_status"] = "not_started"
    st.session_state["chunk_count"] = 0
    st.session_state["embedding_count"] = 0
    st.session_state["indexed_count"] = 0
    st.session_state["embedding_model"] = None
    st.session_state["collection_name"] = None
    st.session_state["ask_code_history"] = []


def save_project_result(
    result: Dict[str, Any],
    source_type: str,
) -> None:
    """
    Save the project details returned by FastAPI.

    The project ID is later used for:
    - /parse-code
    - /index-code
    - RAG retrieval
    - code explanation
    - documentation generation
    """

    st.session_state["project_id"] = result.get(
        "project_id"
    )

    st.session_state["file_count"] = result.get(
        "file_count",
        0,
    )

    st.session_state["project_status"] = result.get(
        "status",
        "unknown",
    )

    st.session_state["source_type"] = source_type

    reset_processing_state()


def display_ingestion_response(
    response: requests.Response,
    source_type: str,
) -> None:
    """
    Display success or error information returned by FastAPI.
    """

    result = get_response_json(response)

    if response.ok:
        save_project_result(
            result=result,
            source_type=source_type,
        )

        st.success("Project ingested successfully.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Status",
                result.get("status", "unknown"),
            )

        with col2:
            st.metric(
                "Files found",
                result.get("file_count", 0),
            )

        with col3:
            st.metric(
                "Source type",
                source_type,
            )

        st.write("Project ID")

        st.code(
            result.get("project_id", ""),
            language=None,
        )

        with st.expander("View complete API response"):
            st.json(result)

    else:
        detail = result.get(
            "detail",
            "The ingestion request failed.",
        )

        st.error(f"Backend error: {detail}")


def ingest_local_project(
    local_path: str,
) -> requests.Response:
    """
    Call POST /ingest-local.
    """

    return requests.post(
        f"{FASTAPI_BASE_URL}/ingest-local",
        json={
            "path": local_path,
        },
        timeout=180,
    )


def ingest_github_project(
    repository_url: str,
    branch: str,
) -> requests.Response:
    """
    Call POST /ingest-github.
    """

    request_body = {
        "repo_url": repository_url,
    }

    if branch:
        request_body["branch"] = branch

    return requests.post(
        f"{FASTAPI_BASE_URL}/ingest-github",
        json=request_body,
        timeout=300,
    )


def ingest_uploaded_zip(
    uploaded_file: Any,
) -> requests.Response:
    """
    Call POST /ingest-upload using multipart form data.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/zip",
        )
    }

    return requests.post(
        f"{FASTAPI_BASE_URL}/ingest-upload",
        files=files,
        timeout=300,
    )


def parse_current_project(
    project_id: str,
) -> requests.Response:
    """
    Call POST /parse-code.
    """

    return requests.post(
        f"{FASTAPI_BASE_URL}/parse-code",
        json={
            "project_id": project_id,
        },
        timeout=300,
    )


def index_current_project(
    project_id: str,
) -> requests.Response:
    """
    Call POST /index-code.

    The backend performs:
    - code chunking
    - Ollama embedding generation
    - ChromaDB indexing
    """

    return requests.post(
        f"{FASTAPI_BASE_URL}/index-code",
        json={
            "project_id": project_id,
        },
        timeout=900,
    )


# ---------------------------------------------------------
# Page title
# ---------------------------------------------------------

st.title("Intelligent Code Documentation Assistant")

st.write(
    "Local RAG and Agentic AI system for code explanation, "
    "Q&A, and documentation generation."
)


# ---------------------------------------------------------
# Backend health check
# ---------------------------------------------------------

st.header("Backend Connection")

if st.button("Check Backend Health"):
    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:
            st.success(
                "FastAPI backend is running successfully."
            )
            st.json(response.json())

        else:
            st.error(
                "The backend responded, but the health "
                "check was unsuccessful."
            )
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. "
            "Make sure the backend is running on port 8000."
        )

    except requests.exceptions.Timeout:
        st.error("Backend request timed out.")

    except requests.RequestException as error:
        st.error(f"Backend request failed: {error}")


st.divider()


# ---------------------------------------------------------
# Source-code input
# ---------------------------------------------------------

st.header("Source Code Input")

input_type = st.radio(
    "Choose how you want to provide source code:",
    [
        "Local Folder Path",
        "GitHub Repository URL",
        "Upload ZIP File",
    ],
)


# ---------------------------------------------------------
# Option 1: Local folder
# ---------------------------------------------------------

if input_type == "Local Folder Path":
    st.subheader("Local Folder Path")

    local_path = st.text_input(
        "Enter local project folder path:",
        placeholder="/path/to/your/project-folder",
    )

    st.caption(
        "The folder must exist on the same computer "
        "where the FastAPI backend is running."
    )

    if st.button(
        "Submit Local Folder",
        type="primary",
    ):
        cleaned_path = local_path.strip()

        if not cleaned_path:
            st.warning(
                "Please enter a valid local folder path."
            )

        else:
            try:
                with st.spinner(
                    "Copying and scanning the local project..."
                ):
                    response = ingest_local_project(
                        local_path=cleaned_path,
                    )

                display_ingestion_response(
                    response=response,
                    source_type="local",
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The local project ingestion request "
                    "timed out."
                )

            except requests.RequestException as error:
                st.error(
                    f"Local ingestion failed: {error}"
                )


# ---------------------------------------------------------
# Option 2: GitHub repository
# ---------------------------------------------------------

elif input_type == "GitHub Repository URL":
    st.subheader("GitHub Repository URL")

    github_url = st.text_input(
        "Enter GitHub repository URL:",
        placeholder=(
            "https://github.com/username/repository-name.git"
        ),
    )

    branch = st.text_input(
        "Branch name:",
        placeholder="main",
        help=(
            "Optional. Leave this empty to use the "
            "repository's default branch."
        ),
    )

    if st.button(
        "Submit GitHub Repository",
        type="primary",
    ):
        cleaned_url = github_url.strip()
        cleaned_branch = branch.strip()

        if not cleaned_url:
            st.warning(
                "Please enter a GitHub repository URL."
            )

        elif not cleaned_url.startswith(
            "https://github.com/"
        ):
            st.error(
                "Please enter a valid HTTPS GitHub URL."
            )

        else:
            try:
                with st.spinner(
                    "Cloning and scanning the GitHub repository..."
                ):
                    response = ingest_github_project(
                        repository_url=cleaned_url,
                        branch=cleaned_branch,
                    )

                display_ingestion_response(
                    response=response,
                    source_type="github",
                )

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI.")

            except requests.exceptions.Timeout:
                st.error(
                    "The GitHub ingestion request timed out."
                )

            except requests.RequestException as error:
                st.error(
                    f"GitHub ingestion failed: {error}"
                )


# ---------------------------------------------------------
# Option 3: ZIP upload
# ---------------------------------------------------------

elif input_type == "Upload ZIP File":
    st.subheader("Upload Project ZIP")

    uploaded_file = st.file_uploader(
        "Upload the source-code project as a ZIP file:",
        type=["zip"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)

        st.success("ZIP file selected.")
        st.write(f"File name: `{uploaded_file.name}`")
        st.write(f"File size: `{file_size_mb:.2f} MB`")

    if st.button(
        "Submit Uploaded Project",
        type="primary",
    ):
        if uploaded_file is None:
            st.warning("Please select a ZIP file.")

        else:
            try:
                with st.spinner(
                    "Uploading and extracting the project..."
                ):
                    response = ingest_uploaded_zip(
                        uploaded_file=uploaded_file,
                    )

                display_ingestion_response(
                    response=response,
                    source_type="upload",
                )

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI.")

            except requests.exceptions.Timeout:
                st.error(
                    "The file upload request timed out."
                )

            except requests.RequestException as error:
                st.error(f"Upload ingestion failed: {error}")


st.divider()


# ---------------------------------------------------------
# Current project
# ---------------------------------------------------------

st.header("Current Project")

if st.session_state["project_id"]:
    st.success(
        "A project is available for parsing and indexing."
    )

    st.write("Project ID")
    st.code(
        st.session_state["project_id"],
        language=None,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Status",
            st.session_state["project_status"],
        )

    with col2:
        st.metric(
            "File count",
            st.session_state["file_count"],
        )

    with col3:
        st.metric(
            "Source type",
            st.session_state["source_type"],
        )

    if st.session_state["parse_completed"]:
        st.subheader("Parsing Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Python files parsed",
                st.session_state["parsed_file_count"],
            )

        with col2:
            st.metric(
                "Parsing errors",
                st.session_state["parsing_error_count"],
            )

    if st.session_state["indexing_status"] != "not_started":
        st.subheader("Indexing Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Indexing status",
                st.session_state["indexing_status"],
            )

        with col2:
            st.metric(
                "Chunks created",
                st.session_state["chunk_count"],
            )

        with col3:
            st.metric(
                "Vectors indexed",
                st.session_state["indexed_count"],
            )

else:
    st.info(
        "No project has been ingested during this session."
    )


st.divider()


# ---------------------------------------------------------
# Parse project
# ---------------------------------------------------------

st.header("Parse Project")

st.caption(
    "Parse all Python files in the current project's "
    "source directory and generate ast.json."
)

if st.session_state["project_id"]:
    if st.button("Parse Source Code"):
        try:
            with st.spinner(
                "Parsing the project source code..."
            ):
                response = parse_current_project(
                    project_id=st.session_state["project_id"],
                )

            result = get_response_json(response)

            if response.ok:
                st.session_state["project_status"] = (
                    result.get("status", "parsed")
                )

                st.session_state["parsed_file_count"] = (
                    result.get("file_count", 0)
                )

                st.session_state["parsing_error_count"] = (
                    result.get("error_count", 0)
                )

                st.session_state["parse_completed"] = True

                # A newly parsed project must be indexed again.
                st.session_state["indexing_status"] = (
                    "not_started"
                )
                st.session_state["chunk_count"] = 0
                st.session_state["embedding_count"] = 0
                st.session_state["indexed_count"] = 0
                st.session_state["embedding_model"] = None
                st.session_state["collection_name"] = None

                st.success("Project parsed successfully.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Parsing status",
                        result.get("status", "parsed"),
                    )

                with col2:
                    st.metric(
                        "Python files parsed",
                        result.get("file_count", 0),
                    )

                with col3:
                    st.metric(
                        "Parsing errors",
                        result.get("error_count", 0),
                    )

                with st.expander(
                    "View parse API response"
                ):
                    st.json(result)

            else:
                st.session_state["parse_completed"] = False

                st.error(
                    result.get(
                        "detail",
                        "Project parsing failed.",
                    )
                )

        except requests.exceptions.ConnectionError:
            st.session_state["parse_completed"] = False
            st.error("Could not connect to FastAPI.")

        except requests.exceptions.Timeout:
            st.session_state["parse_completed"] = False
            st.error("The parsing request timed out.")

        except requests.RequestException as error:
            st.session_state["parse_completed"] = False
            st.error(f"Parsing request failed: {error}")

else:
    st.button(
        "Parse Source Code",
        disabled=True,
        help="Ingest a project first.",
    )


st.divider()


# ---------------------------------------------------------
# Index project
# ---------------------------------------------------------

st.header("Index Project")

st.caption(
    "Create structured code chunks, generate embeddings "
    "with Ollama, and store the vectors in ChromaDB."
)

current_project_id = st.session_state["project_id"]
parse_completed = st.session_state["parse_completed"]

if current_project_id and parse_completed:
    st.success(
        "The project has been parsed and is ready for indexing."
    )

    if st.button(
        "Index Project",
        type="primary",
    ):
        try:
            st.session_state["indexing_status"] = (
                "processing"
            )

            with st.spinner(
                "Chunking source code, generating embeddings, "
                "and indexing vectors in ChromaDB..."
            ):
                response = index_current_project(
                    project_id=current_project_id,
                )

            result = get_response_json(response)

            if response.ok:
                st.session_state["project_status"] = (
                    result.get("status", "indexed")
                )

                st.session_state["indexing_status"] = (
                    result.get("status", "indexed")
                )

                st.session_state["chunk_count"] = (
                    result.get("chunk_count", 0)
                )

                st.session_state["embedding_count"] = (
                    result.get("embedding_count", 0)
                )

                st.session_state["indexed_count"] = (
                    result.get("indexed_count", 0)
                )

                st.session_state["embedding_model"] = (
                    result.get("embedding_model")
                )

                st.session_state["collection_name"] = (
                    result.get("collection_name")
                )

                st.success(
                    "Project indexed successfully."
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Status",
                        result.get(
                            "status",
                            "indexed",
                        ),
                    )

                with col2:
                    st.metric(
                        "Files processed",
                        result.get(
                            "file_count",
                            0,
                        ),
                    )

                with col3:
                    st.metric(
                        "Chunks created",
                        result.get(
                            "chunk_count",
                            0,
                        ),
                    )

                with col4:
                    st.metric(
                        "Vectors indexed",
                        result.get(
                            "indexed_count",
                            0,
                        ),
                    )

                details_col1, details_col2 = st.columns(2)

                with details_col1:
                    st.write("Embedding model")
                    st.code(
                        result.get(
                            "embedding_model",
                            "",
                        ),
                        language=None,
                    )

                with details_col2:
                    st.write("ChromaDB collection")
                    st.code(
                        result.get(
                            "collection_name",
                            "",
                        ),
                        language=None,
                    )

                with st.expander(
                    "View indexing API response"
                ):
                    st.json(result)

            else:
                st.session_state["indexing_status"] = (
                    "failed"
                )

                st.error(
                    result.get(
                        "detail",
                        "Project indexing failed.",
                    )
                )

        except requests.exceptions.ConnectionError:
            st.session_state["indexing_status"] = (
                "failed"
            )

            st.error(
                "Could not connect to FastAPI. "
                "Make sure the backend is running."
            )

        except requests.exceptions.Timeout:
            st.session_state["indexing_status"] = (
                "failed"
            )

            st.error(
                "The indexing request timed out. "
                "A large project or local embedding model "
                "may require additional processing time."
            )

        except requests.RequestException as error:
            st.session_state["indexing_status"] = (
                "failed"
            )

            st.error(
                f"Indexing request failed: {error}"
            )

elif current_project_id:
    st.warning(
        "Parse the current project before indexing it."
    )

    st.button(
        "Index Project",
        disabled=True,
        help="Parse the source code first.",
    )

else:
    st.info(
        "Ingest and parse a project before indexing."
    )

    st.button(
        "Index Project",
        disabled=True,
        help="Ingest a project first.",
    )


st.divider()



# ---------------------------------------------------------
# Ask Code
# ---------------------------------------------------------

render_ask_code_section(
    FASTAPI_BASE_URL
)


st.divider()
# ---------------------------------------------------------
# Future features
# ---------------------------------------------------------

st.header("Future Modes")

col1, col2 = st.columns(2)

with col1:
    if st.button("Explain Code"):
        st.info(
            "Explain Code mode will be connected "
            "in a later sprint."
        )

with col2:
    if st.button("Generate Docs"):
        st.info(
            "Generate Docs mode will be connected "
            "in a later sprint."
        )