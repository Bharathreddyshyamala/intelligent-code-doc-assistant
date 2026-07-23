# Intelligent Code Documentation Assistant

An AI-powered application that parses source code using AST, creates code chunks, generates embeddings with Ollama, stores vectors in ChromaDB, and answers questions about the indexed codebase using RAG.

---

## Features

- AST-based Python code parsing
- Function and class-level code chunking
- Local embeddings using Ollama
- Persistent vector storage using ChromaDB
- Semantic code retrieval
- Ask Code using a local chat model
- Source references with file names and line numbers
- FastAPI backend
- Streamlit frontend

---

# Local Setup Instructions

## Prerequisites

Install the following before running the project:

- Git
- Python 3.9 or later
- Ollama
- A terminal or command prompt
- VS Code or another code editor

---

## 1. Clone the GitHub Repository

Open a terminal and run:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd intelligent-code-doc-assistant
```

Switch to the main branch and download the latest code:

```bash
git checkout main
git pull origin main
```

---

## 2. Create a Python Virtual Environment

Move into the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python3 -m venv venv
```

### Activate on macOS or Linux

```bash
source venv/bin/activate
```

### Activate on Windows Command Prompt

```cmd
venv\Scripts\activate
```

### Activate on Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(venv)
```

---

## 3. Install Project Dependencies

Make sure the virtual environment is active and you are inside the `backend` directory.

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install all required packages:

```bash
python -m pip install -r requirements.txt
```

Verify important packages:

```bash
python -m pip show fastapi
python -m pip show chromadb
python -m pip show ollama
python -m pip show streamlit
```

---

## 4. Configure Environment Variables

Inside the `backend` directory, create a local `.env` file from `.env.example`.

### macOS or Linux

```bash
cp .env.example .env
```

### Windows Command Prompt

```cmd
copy .env.example .env
```

The `.env` file should contain:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434

OLLAMA_EMBED_MODEL=embeddinggemma
EMBEDDING_BATCH_SIZE=32

OLLAMA_CHAT_MODEL=qwen2.5-coder:3b
OLLAMA_CHAT_TEMPERATURE=0.2

CHROMA_DB_PATH=vector_store/chroma_db
CHROMA_COLLECTION_NAME=code_chunks
CHROMA_BATCH_SIZE=100

RAG_MAX_CONTEXT_CHARS=24000
RAG_MAX_CHUNK_CONTEXT_CHARS=5000
RAG_SOURCE_EXCERPT_CHARS=1200
```

Do not commit the `.env` file to GitHub.

The `.env.example` file can be committed because it should not contain passwords or secret keys.

---

## 5. Install Ollama Models

Verify that Ollama is installed:

```bash
ollama --version
```

Download the embedding model:

```bash
ollama pull embeddinggemma
```

Download the code chat model:

```bash
ollama pull qwen2.5-coder:3b
```

Verify the installed models:

```bash
ollama list
```

The output should contain:

```text
embeddinggemma
qwen2.5-coder:3b
```

Test whether the Ollama server is running:

```bash
curl http://127.0.0.1:11434/api/tags
```

When Ollama is not running, start it with:

```bash
ollama serve
```

Keep the Ollama terminal open.

On macOS, opening the Ollama application may start the server automatically.

---

## 6. Start the FastAPI Backend

Open a terminal and move into the backend directory:

```bash
cd intelligent-code-doc-assistant/backend
```

Activate the virtual environment.

### macOS or Linux

```bash
source venv/bin/activate
```

### Windows

```cmd
venv\Scripts\activate
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Keep this terminal running.

---

## 7. Start the Streamlit Frontend

Open another terminal and move into the project root:

```bash
cd intelligent-code-doc-assistant
```

Activate the backend virtual environment.

### macOS or Linux

```bash
source backend/venv/bin/activate
```

### Windows

```cmd
backend\venv\Scripts\activate
```

Start Streamlit:

```bash
streamlit run frontend/streamlit_app.py
```

The frontend will normally open at:

```text
http://localhost:8501
```

---

## 8. Application Workflow

Use the application in the following order:

```text
1. Ingest Project
2. Parse Source Code
3. Index Project
4. Ask Code Questions
```

### Step 1: Ingest Project

Provide source code using one of the available options:

- Local folder path
- GitHub repository URL
- ZIP file upload

### Step 2: Parse Source Code

The parser reads Python files using AST and creates:

```text
ast.json
```

### Step 3: Index Project

The indexing workflow is:

```text
Parsed source code
→ Code chunks
→ Ollama embeddings
→ ChromaDB storage
```

The `embeddinggemma` model converts source-code chunks into embedding vectors.

The ChromaDB database is created automatically inside:

```text
backend/vector_store/chroma_db/
```

This folder is ignored by Git and should not be committed.

### Step 4: Ask Code

After indexing, ask questions such as:

```text
How is division by zero handled?
```

```text
Where is user input validated?
```

```text
Which function displays the result?
```

The Ask Code workflow is:

```text
User question
→ Query embedding
→ ChromaDB similarity search
→ Relevant source-code chunks
→ qwen2.5-coder chat model
→ Answer with source references
```

---

## 9. Inspect ChromaDB Records

Move into the backend directory:

```bash
cd backend
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

View all stored ChromaDB records:

```bash
python inspect_chroma.py
```

View records for one project:

```bash
python inspect_chroma.py --project-id <PROJECT_ID>
```

View only a limited number of records:

```bash
python inspect_chroma.py --limit 10
```

Export records to a readable JSON file:

```bash
python inspect_chroma.py \
  --project-id <PROJECT_ID> \
  --export-json
```

---

## 10. Stop the Application

To stop FastAPI, Streamlit, or Ollama, open the corresponding terminal and press:

```text
Control + C
```

Deactivate the virtual environment:

```bash
deactivate
```

---

# Updating an Existing Local Repository

Move into the project:

```bash
cd intelligent-code-doc-assistant
```

Switch to the main branch:

```bash
git checkout main
```

Download the latest code:

```bash
git pull origin main
```

Activate the virtual environment:

```bash
source backend/venv/bin/activate
```

Install any newly added dependencies:

```bash
python -m pip install -r backend/requirements.txt
```

---

# Contributor Workflow

Before starting new work:

```bash
git checkout main
git pull origin main
```

Create a new feature branch:

```bash
git switch -c feature/<feature-name>
```

Example:

```bash
git switch -c feature/document-generation
```

After completing the feature:

```bash
git status
git add .
git commit -m "Add document generation feature"
git push -u origin feature/document-generation
```

Create a pull request from the feature branch into `main`.

---

# Troubleshooting

## Ollama Command Not Found

Verify Ollama:

```bash
ollama --version
```

Install Ollama when the command is unavailable, then restart the terminal.

---

## Ollama Model Not Found

Download the required models:

```bash
ollama pull embeddinggemma
ollama pull qwen2.5-coder:3b
```

---

## Cannot Connect to Ollama

Start the Ollama server:

```bash
ollama serve
```

Test the server:

```bash
curl http://127.0.0.1:11434/api/tags
```

---

## Streamlit Command Not Found

Activate the virtual environment:

```bash
source backend/venv/bin/activate
```

Install the requirements:

```bash
python -m pip install -r backend/requirements.txt
```

Start Streamlit again:

```bash
streamlit run frontend/streamlit_app.py
```

---

## FastAPI Port Is Already in Use

Find the process using port `8000`:

```bash
lsof -i :8000
```

Stop the process:

```bash
kill -9 <PROCESS_ID>
```

Alternatively, start FastAPI on another port:

```bash
uvicorn main:app --reload --port 8001
```

---

## Ask Code Is Unavailable

Make sure the project completed all required stages:

```text
Ingested
→ Parsed
→ Indexed
```

Verify both Ollama models:

```bash
ollama list
```

---

## ChromaDB Contains No Records

Index a project from Streamlit, then run:

```bash
cd backend
python inspect_chroma.py
```
