from typing import Any, Dict

import requests
import streamlit as st


def get_response_json(
    response: requests.Response,
) -> Dict[str, Any]:
    """
    Safely convert a backend response into JSON.
    """

    try:
        return response.json()

    except ValueError:
        return {
            "detail": (
                "The backend returned an invalid response."
            )
        }


def ask_code_api(
    fastapi_base_url: str,
    project_id: str,
    question: str,
    top_k: int,
) -> requests.Response:
    """
    Call POST /ask-code.
    """

    return requests.post(
        f"{fastapi_base_url}/ask-code",
        json={
            "project_id": project_id,
            "question": question,
            "top_k": top_k,
        },
        timeout=900,
    )


def display_sources(
    sources: list,
) -> None:
    """
    Display source references returned by the RAG API.
    """

    if not sources:
        st.warning(
            "The answer did not include source references."
        )
        return

    for source in sources:
        source_number = source.get(
            "source_number",
            "",
        )

        file_path = source.get(
            "file_path",
            "unknown",
        )

        symbol = source.get(
            "symbol",
            "unknown",
        )

        start_line = source.get(
            "start_line",
            "",
        )

        end_line = source.get(
            "end_line",
            "",
        )

        title = (
            f"Source {source_number}: "
            f"{file_path} — {symbol}"
        )

        with st.expander(title):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(
                    f"**Type:** "
                    f"{source.get('source_type', '')}"
                )

            with col2:
                st.write(
                    f"**Lines:** "
                    f"{start_line}-{end_line}"
                )

            with col3:
                st.write(
                    f"**Distance:** "
                    f"{source.get('distance', '')}"
                )

            st.code(
                source.get(
                    "excerpt",
                    "",
                ),
                language=None,
            )


def render_ask_code_section(
    fastapi_base_url: str,
) -> None:
    """
    Render the complete Ask Code Streamlit interface.
    """

    st.header("Ask Code")

    st.caption(
        "Ask questions about the current indexed project. "
        "The system retrieves relevant code from ChromaDB "
        "and generates an answer using Ollama."
    )

    if "ask_code_history" not in st.session_state:
        st.session_state["ask_code_history"] = []

    project_id = st.session_state.get(
        "project_id"
    )

    indexing_status = st.session_state.get(
        "indexing_status",
        "not_started",
    )

    if not project_id:
        st.info(
            "Ingest, parse, and index a project before "
            "asking code questions."
        )
        return

    if indexing_status != "indexed":
        st.warning(
            "The current project must be indexed before "
            "Ask Code can be used."
        )
        return

    with st.form(
        "ask_code_form",
        clear_on_submit=False,
    ):
        question = st.text_area(
            "Enter your question:",
            placeholder=(
                "Example: How does the calculator handle "
                "division by zero?"
            ),
            height=120,
        )

        top_k = st.slider(
            "Number of code chunks to retrieve",
            min_value=1,
            max_value=20,
            value=8,
            help=(
                "A larger value provides more context but "
                "also creates a larger prompt."
            ),
        )

        submitted = st.form_submit_button(
            "Ask Code",
            type="primary",
        )

    if submitted:
        cleaned_question = question.strip()

        if not cleaned_question:
            st.warning(
                "Please enter a question."
            )

        else:
            try:
                with st.spinner(
                    "Retrieving source code and generating "
                    "an answer..."
                ):
                    response = ask_code_api(
                        fastapi_base_url=(
                            fastapi_base_url
                        ),
                        project_id=project_id,
                        question=cleaned_question,
                        top_k=top_k,
                    )

                result = get_response_json(
                    response
                )

                if response.ok:
                    st.session_state[
                        "ask_code_history"
                    ].append(
                        {
                            "question": cleaned_question,
                            "answer": result.get(
                                "answer",
                                "",
                            ),
                            "chat_model": result.get(
                                "chat_model",
                                "",
                            ),
                            "retrieved_chunk_count": (
                                result.get(
                                    "retrieved_chunk_count",
                                    0,
                                )
                            ),
                            "used_chunk_count": (
                                result.get(
                                    "used_chunk_count",
                                    0,
                                )
                            ),
                            "sources": result.get(
                                "sources",
                                [],
                            ),
                        }
                    )

                    st.success(
                        "Answer generated successfully."
                    )

                else:
                    st.error(
                        result.get(
                            "detail",
                            "Ask Code request failed.",
                        )
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to FastAPI."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The Ask Code request timed out. "
                    "The local chat model may require "
                    "additional processing time."
                )

            except requests.RequestException as error:
                st.error(
                    f"Ask Code request failed: {error}"
                )

    history = st.session_state[
        "ask_code_history"
    ]

    if history:
        st.subheader("Conversation")

        for item in history:
            with st.chat_message("user"):
                st.write(
                    item["question"]
                )

            with st.chat_message("assistant"):
                st.markdown(
                    item["answer"]
                )

                info_col1, info_col2, info_col3 = (
                    st.columns(3)
                )

                with info_col1:
                    st.caption(
                        "Model: "
                        f"{item['chat_model']}"
                    )

                with info_col2:
                    st.caption(
                        "Retrieved chunks: "
                        f"{item['retrieved_chunk_count']}"
                    )

                with info_col3:
                    st.caption(
                        "Used chunks: "
                        f"{item['used_chunk_count']}"
                    )

                display_sources(
                    item["sources"]
                )

        if st.button(
            "Clear Ask Code History",
            key="clear_ask_code_history",
        ):
            st.session_state[
                "ask_code_history"
            ] = []

            st.rerun()