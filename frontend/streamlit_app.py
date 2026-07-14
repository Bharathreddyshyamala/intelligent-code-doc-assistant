import streamlit as st
import requests


FASTAPI_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Intelligent Code Documentation Assistant",
    page_icon="📘",
    layout="wide"
)


st.title(" Intelligent Code Documentation Assistant")
st.write("Local RAG and Agentic AI system for code explanation, Q&A, and documentation generation.")



# Backend Health Check


st.header("Backend Connection")

if st.button("Check Backend Health"):
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=5)

        if response.status_code == 200:
            st.success("FastAPI backend is running successfully.")
            st.json(response.json())
        else:
            st.error("Backend responded, but something went wrong.")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Make sure backend is running on port 8000.")

    except requests.exceptions.Timeout:
        st.error("Backend request timed out.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")


st.divider()



# Source Code Input Section


st.header("Source Code Input")

input_type = st.radio(
    "Choose how you want to provide source code:",
    [
        "Local Folder Path",
        "GitHub Repository URL",
        "Upload Code Files"
    ]
)



# Option 1: Local Folder Path


if input_type == "Local Folder Path":
    st.subheader("Local Folder Path")

    local_path = st.text_input(
        "Enter local project folder path:",
        placeholder="/Users/yourname/Desktop/sample-project"
    )

    if st.button("Submit Local Folder"):
        if local_path.strip():
            st.success("Local folder path received.")
            st.write("Path:", local_path)

            st.info("Later, this will call FastAPI endpoint like /ingest-local.")
        else:
            st.warning("Please enter a valid local folder path.")



# Option 2: GitHub Repository URL


elif input_type == "GitHub Repository URL":
    st.subheader("GitHub Repository URL")

    github_url = st.text_input(
        "Enter GitHub repository URL:",
        placeholder="https://github.com/username/repository-name"
    )

    if st.button("Submit GitHub Repository"):
        if github_url.strip():
            if github_url.startswith("https://github.com/"):
                st.success("GitHub repository URL received.")
                st.write("Repository URL:", github_url)

                st.info("Later, this will call FastAPI endpoint like /ingest-github.")
            else:
                st.error("Please enter a valid GitHub repository URL.")
        else:
            st.warning("Please enter a GitHub repository URL.")



# Option 3: File Upload


elif input_type == "Upload Code Files":
    st.subheader("Upload Code Files")

    uploaded_files = st.file_uploader(
        "Upload source code files",
        type=["py", "js", "java", "cpp", "c", "ts", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

        for uploaded_file in uploaded_files:
            st.write(f"File name: {uploaded_file.name}")

            file_content = uploaded_file.read().decode("utf-8", errors="ignore")

            with st.expander(f"Preview: {uploaded_file.name}"):
                st.code(file_content[:3000])

        if st.button("Submit Uploaded Files"):
            st.success("Uploaded files received.")
            st.info("Later, this will call FastAPI endpoint like /ingest-upload.")
    else:
        st.info("Upload one or more source code files.")


st.divider()



# Future Features Placeholder


st.header("Future Modes")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Explain Code"):
        st.info("Explain Code mode will be connected in a later sprint.")

with col2:
    if st.button("Ask Code"):
        st.info("Ask Code mode will be connected in a later sprint.")

with col3:
    if st.button("Generate Docs"):
        st.info("Generate Docs mode will be connected in a later sprint.")