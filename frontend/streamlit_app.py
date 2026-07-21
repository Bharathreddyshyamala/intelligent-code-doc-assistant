import os
from typing import Any, Dict

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


def save_project_result(
    result: Dict[str, Any],
    source_type: str,
) -> None:
    """
    Save the project details returned by FastAPI.

    The project ID is later used for:
    - /parse-code
    - /chunk-code
    - RAG indexing
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
        "A project is available for parsing and chunking."
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
                response = requests.post(
                    f"{FASTAPI_BASE_URL}/parse-code",
                    json={
                        "project_id": st.session_state[
                            "project_id"
                        ]
                    },
                    timeout=300,
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
                st.error(
                    result.get(
                        "detail",
                        "Project parsing failed.",
                    )
                )

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI.")

        except requests.exceptions.Timeout:
            st.error("The parsing request timed out.")

        except requests.RequestException as error:
            st.error(f"Parsing request failed: {error}")

else:
    st.button(
        "Parse Source Code",
        disabled=True,
        help="Ingest a project first.",
    )


st.divider()


# ---------------------------------------------------------
# Future features
# ---------------------------------------------------------
st.header("Future Modes")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Explain Code"):
        st.info(
            "Explain Code mode will be connected "
            "in a later sprint."
        )

with col2:
    if st.button("Ask Code"):
        st.info(
            "Ask Code mode will be connected "
            "in a later sprint."
        )

with col3:
    if st.button("Generate Docs"):
        st.info(
            "Generate Docs mode will be connected "
            "in a later sprint."
        )