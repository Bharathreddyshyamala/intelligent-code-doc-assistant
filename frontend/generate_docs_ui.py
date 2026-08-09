from typing import Any, Dict

import requests
import streamlit as st


def _get_response_json(
    response: requests.Response,
) -> Dict[str, Any]:
    """Safely read a FastAPI JSON response."""
    try:
        return response.json()
    except ValueError:
        return {
            "detail": "The backend returned an invalid response."
        }


def _initialise_documentation_state() -> None:
    """Initialise Streamlit state used by Generate Docs."""
    defaults = {
        "docs_project_id": None,
        "docs_generation_status": "not_started",
        "docs_result": None,
        "generated_pdf_bytes": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _reset_docs_for_new_project(
    project_id: str,
) -> None:
    """Clear generated documentation when the selected project changes."""
    previous_project_id = st.session_state.get(
        "docs_project_id"
    )

    if previous_project_id != project_id:
        st.session_state["docs_project_id"] = project_id
        st.session_state["docs_generation_status"] = "not_started"
        st.session_state["docs_result"] = None
        st.session_state["generated_pdf_bytes"] = None



def _create_pdf_filename(
    project_title: str,
) -> str:
    """
    Create a safe project-specific PDF download filename.
    """

    safe_title = "".join(
        character
        if character.isalnum()
        or character in {
            "-",
            "_",
            " ",
        }
        else "_"
        for character in project_title
    )

    safe_title = "_".join(
        safe_title.split()
    ).strip("_")

    if not safe_title:
        safe_title = "Project"

    return "{}_Documentation.pdf".format(
        safe_title
    )

def render_generate_docs_ui(
    fastapi_base_url: str,
) -> None:
    """
    Render the Generate Documentation section.

    fastapi_base_url should include /api/v1 in the current project setup.
    """
    _initialise_documentation_state()

    st.header("Generate Documentation")

    st.caption(
        "Generate project-level, file-level, class-level, and "
        "function-level documentation as a downloadable PDF."
    )

    project_id = st.session_state.get("project_id")
    parse_completed = st.session_state.get(
        "parse_completed",
        False,
    )

    if not project_id:
        st.info(
            "Ingest a project before generating documentation."
        )
        st.button(
            "Generate PDF Documentation",
            disabled=True,
            help="No project is currently selected.",
        )
        return

    _reset_docs_for_new_project(project_id)

    if not parse_completed:
        st.warning(
            "Parse the current project before generating documentation. "
            "Generate Docs reads ast.json."
        )
        st.button(
            "Generate PDF Documentation",
            disabled=True,
            help="Parse the project first.",
        )
        return

    st.success(
        "The current project is ready for documentation generation."
    )

    st.write("Project ID")
    st.code(project_id, language=None)

    detected_project_name = (
        st.session_state.get("project_name")
        or "Project {}".format(project_id[:8])
    )

    project_title = st.text_input(
        "Project title",
        value=detected_project_name,
        key="documentation_title_{}".format(project_id),
        help=(
            "The title is detected from the local folder, "
            "GitHub repository, or uploaded ZIP filename. "
            "You can edit it before generating the PDF."
        ),
    )

    maximum_files = st.number_input(
        "Maximum source files to include",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        help=(
            "The generator makes approximately one Ollama request per file. "
            "Lower values generate faster. Files beyond this limit are not "
            "included in this documentation run."
        ),
    )

    overwrite = st.checkbox(
        "Regenerate existing documentation",
        value=True,
    )

    if st.button(
        "Generate PDF Documentation",
        type="primary",
        use_container_width=True,
    ):
        cleaned_title = project_title.strip()

        if not cleaned_title:
            st.warning("Enter a project title.")
        else:
            st.session_state["docs_generation_status"] = "generating"
            generation_url = (
                "{}/generate-docs".format(
                    fastapi_base_url.rstrip("/")
                )
            )

            try:
                with st.spinner(
                    "Generating documentation and creating the PDF..."
                ):
                    response = requests.post(
                        generation_url,
                        json={
                            "project_id": project_id,
                            "project_title": cleaned_title,
                            "max_files": int(maximum_files),
                            "overwrite": overwrite,
                        },
                        timeout=1800,
                    )

                result = _get_response_json(response)

                if not response.ok:
                    st.session_state["docs_generation_status"] = "failed"

                    st.error(
                        "Documentation request failed.\n\n"
                        "Status: {}\n\n"
                        "URL: {}\n\n"
                        "Backend response: {}".format(
                            response.status_code,
                            generation_url,
                            result.get(
                                "detail",
                                "Documentation generation failed.",
                            ),
                        )
                    )
                    return

                download_url = (
                    "{}/generated-docs/{}/download".format(
                        fastapi_base_url.rstrip("/"),
                        project_id,
                    )
                )

                pdf_response = requests.get(
                    download_url,
                    timeout=300,
                )

                if not pdf_response.ok:
                    st.session_state["docs_generation_status"] = "failed"
                    st.error(
                        "The PDF was generated, but it could not be downloaded. "
                        "Backend status: {}".format(pdf_response.status_code)
                    )
                    return

                result["project_title"] = (
                    result.get("project_title")
                    or cleaned_title
                )

                st.session_state["docs_generation_status"] = "completed"
                st.session_state["docs_result"] = result
                st.session_state["generated_pdf_bytes"] = pdf_response.content

                st.success(
                    "PDF documentation generated successfully."
                )

            except requests.exceptions.ConnectionError:
                st.session_state["docs_generation_status"] = "failed"
                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.Timeout:
                st.session_state["docs_generation_status"] = "failed"
                st.error(
                    "Documentation generation timed out. "
                    "Try generating documentation for fewer files."
                )

            except requests.RequestException as error:
                st.session_state["docs_generation_status"] = "failed"
                st.error(
                    "Documentation request failed: {}".format(error)
                )

    docs_result = st.session_state.get("docs_result")
    generated_pdf_bytes = st.session_state.get(
        "generated_pdf_bytes"
    )

    if (
        docs_result
        and generated_pdf_bytes
        and st.session_state.get("docs_generation_status") == "completed"
    ):
        st.subheader("Documentation Result")

        column1, column2, column3 = st.columns(3)

        with column1:
            st.metric(
                "Status",
                docs_result.get("status", "completed"),
            )

        with column2:
            st.metric(
                "Documented files",
                docs_result.get("documented_file_count", 0),
            )

        with column3:
            st.metric(
                "Skipped files",
                docs_result.get("skipped_file_count", 0),
            )

        st.write("Generated PDF")
        st.code(
            docs_result.get("output_path", ""),
            language=None,
        )

        st.download_button(
            label="Download PDF Documentation",
            data=generated_pdf_bytes,
            file_name=_create_pdf_filename(
                docs_result.get(
                    "project_title",
                    detected_project_name,
                )
            ),
            mime="application/pdf",
            use_container_width=True,
        )

        preview_url = (
            "{}/generated-docs/{}".format(
                fastapi_base_url.rstrip("/"),
                project_id,
            )
        )

        st.markdown(
            "[Open PDF in a new browser tab]({})".format(
                preview_url
            )
        )

        with st.expander(
            "View Generate Docs API Response"
        ):
            st.json(docs_result)