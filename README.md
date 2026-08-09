# Intelligent Code Documentation Assistant

A **local AI-powered source-code understanding and documentation platform** that can ingest a software project, parse Python source code using AST, create structured code chunks, generate local embeddings with Ollama, index them in ChromaDB, answer questions about the codebase using RAG, and automatically generate technical documentation as a PDF.

The application is designed to run locally so source code, embeddings, vector storage, LLM inference, and generated documentation can remain on the developer's machine.

---

## Table of Contents

- [Overview](#overview)
- [Current Features](#current-features)
- [System Architecture](#system-architecture)
- [End-to-End Workflow](#end-to-end-workflow)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Installation](#installation)
- [Ollama Setup](#ollama-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [API Overview](#api-overview)
- [Generated Project Data](#generated-project-data)
- [RAG Pipeline](#rag-pipeline)
- [Documentation Generation](#documentation-generation)
- [Example Workflow](#example-workflow)
- [Local-First Design](#local-first-design)
- [Troubleshooting](#troubleshooting)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Understanding an unfamiliar codebase can require significant time. Developers often need to inspect files manually, identify classes and functions, trace dependencies, understand execution flow, search for important logic, and create documentation.

The **Intelligent Code Documentation Assistant** automates much of that workflow.

A user can provide source code using:

- A local project folder
- A GitHub repository URL
- A ZIP file upload

The system then processes the project through a structured pipeline:

```text
Source Project
     |
     v
Project Ingestion
     |
     v
AST Parsing
     |
     v
ast.json
     |
     v
Code Chunking
     |
     v
chunks.json
     |
     v
Ollama Embeddings
     |
     v
ChromaDB Vector Store
     |
     +-------------------------+
     |                         |
     v                         v
Ask Code / RAG          Generate Documentation
     |                         |
     v                         v
Relevant Code           AST + Source Code
     |                         |
     v                         v
Ollama LLM              Ollama LLM
     |                         |
     v                         v
Grounded Answer         PDF Documentation
```

The project combines **deterministic static code analysis** with **local generative AI**.

---

## Current Features

### Source-code ingestion

The application supports:

- **Local Folder Path**
- **GitHub Repository URL**
- **ZIP File Upload**

Each ingested project receives a unique `project_id`. That ID is reused for parsing, indexing, retrieval, and documentation generation.

### AST-based Python parsing

The parser uses Python's Abstract Syntax Tree (`ast`) module to extract structured code information.

The parser can capture information such as:

- Files and modules
- Imports
- Module variables
- Classes
- Base classes
- Class attributes
- Instance attributes
- Methods
- Functions
- Parameters
- Type annotations
- Default parameter values
- Decorators
- Docstrings
- Return statements
- Function calls
- Raised exceptions
- Handled exceptions
- Conditions
- Loops
- Try blocks
- Source-code line ranges
- Symbol source code
- Internal dependency information

The structured project output is stored in:

```text
ast.json
```

### Structured code chunking

The AST output is transformed into smaller semantic code units suitable for embedding and retrieval.

Chunks can represent:

- Functions
- Methods
- Classes
- Class summaries
- Module-level information
- Logical code sections

Chunk metadata can preserve fields such as:

```text
project_id
file_path
module_name
symbol_name
qualified_name
chunk_kind
start_line
end_line
docstring
source code
```

The resulting chunks are stored in:

```text
chunks.json
```

### Local embeddings with Ollama

Code chunks are converted into embeddings using a locally running Ollama embedding model.

The project currently uses an embedding model such as:

```text
embeddinggemma
```

### ChromaDB indexing

Embeddings, source content, and metadata are stored in a local ChromaDB collection.

ChromaDB provides the vector-search layer used by the RAG pipeline.

### Ask Code

The **Ask Code** feature allows users to ask natural-language questions about the indexed repository.

Examples:

```text
How does project indexing work?

Where are Ollama embeddings generated?

Explain the flow between the indexing route and indexer service.

What does the AST parser extract?

How is a project ID validated?
```

The system retrieves relevant project code and supplies that context to the local Ollama generation model.

### Automatic documentation generation

The documentation generator:

1. Reads `ast.json`.
2. Reads the original source files.
3. Generates file-level documentation using Ollama.
4. Generates a project-level summary.
5. Converts the generated content into a PDF.
6. Stores the generated PDF locally.

Generated documentation can cover:

- Project purpose
- Architecture
- Main files
- Imports
- Dependencies
- Classes
- Functions
- Methods
- Parameters
- Return values
- Exceptions
- Execution flow
- File responsibilities
- Documentation notes

### Project-specific PDF output

The documentation title is derived from the ingested project name.

Examples:

```text
calculator
        ->
Calculator_Documentation.pdf
```

```text
library-management
        ->
Library_Management_Documentation.pdf
```

### Streamlit frontend

The frontend provides controls for:

- Backend health checking
- Local folder ingestion
- GitHub repository ingestion
- ZIP upload
- Parsing
- Indexing
- Ask Code
- PDF documentation generation
- PDF download

### FastAPI backend

FastAPI provides the API layer for ingestion, parsing, indexing, RAG operations, and documentation generation.

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## System Architecture

```text
+--------------------------------------------------------+
|                  Streamlit Frontend                    |
|                                                        |
| Ingest | Parse | Index | Ask Code | Generate PDF Docs |
+------------------------------+-------------------------+
                               |
                               | HTTP
                               v
+--------------------------------------------------------+
|                     FastAPI Backend                    |
|                                                        |
|                       API Routes                       |
+------------------------------+-------------------------+
                               |
                               v
+--------------------------------------------------------+
|                  Application Services                  |
|                                                        |
| file_scanner.py                                        |
| ast_parser.py                                          |
| chunker.py                                             |
| embedding_service.py                                   |
| indexer.py                                             |
| retrieval_service.py                                   |
| reranker_service.py                                    |
| ollama_service.py                                      |
| rag_service.py                                         |
| doc_generator.py                                       |
+--------------------+------------------+----------------+
                     |                  |
                     v                  v
              +-------------+     +-------------+
              |  ChromaDB   |     |   Ollama    |
              | Vector DB   |     | Local Models|
              +-------------+     +-------------+
```

### Routes vs Services

**Routes** handle the HTTP layer:

- Receive requests
- Validate request bodies
- Call services
- Return responses
- Convert errors into HTTP status codes

**Services** contain application logic:

- File scanning
- AST parsing
- Chunk generation
- Embedding generation
- Indexing
- Retrieval
- Ollama interaction
- RAG
- PDF documentation generation

Example:

```text
POST /index-code
      |
      v
indexing_routes.py
      |
      v
indexer.py
      |
      +--> chunker.py
      +--> embedding_service.py
      +--> ChromaDB
```

---

## End-to-End Workflow

### 1. Ingest

```text
Local Folder
or
GitHub Repository
or
ZIP Upload
       |
       v
Project Workspace
       |
       v
project_id
```

### 2. Parse

```text
Project Source
     |
     v
ast_parser.py
     |
     v
ast.json
```

### 3. Index

```text
ast.json
   |
   v
chunker.py
   |
   v
chunks.json
   |
   v
Ollama Embeddings
   |
   v
ChromaDB
```

### 4. Ask Code

```text
User Question
     |
     v
Question Embedding
     |
     v
Retrieve Relevant Chunks
     |
     v
Build RAG Context
     |
     v
Ollama Code Model
     |
     v
Grounded Answer
```

### 5. Generate Documentation

```text
ast.json
   +
Original Source Files
   |
   v
Documentation Prompts
   |
   v
Ollama
   |
   v
Structured Documentation
   |
   v
ReportLab
   |
   v
<Project>_Documentation.pdf
```

Indexing is required for **Ask Code**, but the current documentation flow can run after parsing because it reads AST data and source code directly.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Static Analysis | Python AST |
| Local LLM Runtime | Ollama |
| Generation Model | `qwen2.5-coder` or configured Ollama model |
| Embedding Model | `embeddinggemma` or configured Ollama embedding model |
| Vector Database | ChromaDB |
| API Server | Uvicorn |
| PDF Generation | ReportLab |
| HTTP Client | Requests |
| Data Storage | JSON + local filesystem + ChromaDB |
| Version Control | Git / GitHub |

---

## Project Structure

A representative repository structure is:

```text
intelligent-code-doc-assistant/
|
+-- backend/
|   |
|   +-- main.py
|   +-- requirements.txt
|   +-- .env
|   +-- .env.example
|   |
|   +-- routes/
|   |   +-- ingestion_routes.py
|   |   +-- parser_routes.py
|   |   +-- indexing_routes.py
|   |   +-- rag_routes.py
|   |   +-- docs_routes.py
|   |   +-- evaluation_routes.py
|   |
|   +-- services/
|   |   +-- github_ingestion.py
|   |   +-- file_scanner.py
|   |   +-- ast_parser.py
|   |   +-- chunker.py
|   |   +-- embedding_service.py
|   |   +-- indexer.py
|   |   +-- retrieval_service.py
|   |   +-- reranker_service.py
|   |   +-- ollama_service.py
|   |   +-- rag_service.py
|   |   +-- doc_generator.py
|   |   +-- langgraph_agent.py
|   |   +-- ragas_evaluator.py
|   |
|   +-- vector_store/
|   +-- temp_repos/
|   +-- generated_docs/
|   +-- uploads/
|
+-- frontend/
|   +-- streamlit_app.py
|   +-- ask_code_ui.py
|   +-- generate_docs_ui.py
|
+-- README.md
+-- .gitignore
```

Some modules shown above can be optional or future-facing depending on the current development branch.

---

## Core Components

### `file_scanner.py`

Responsible for project workspace and file management.

Typical responsibilities include:

- Validate a `project_id`
- Locate source directories
- Locate `ast.json`
- Locate `chunks.json`
- Read project metadata
- Update project metadata
- Manage project-specific paths

### `ast_parser.py`

Converts Python source code into structured program metadata.

Conceptually:

```text
Python Source
    |
    v
ast.parse()
    |
    v
AST Traversal
    |
    v
Structured Project JSON
```

Example source:

```python
def add(a: int, b: int) -> int:
    return a + b
```

Conceptual AST metadata:

```json
{
  "name": "add",
  "parameters": [
    {
      "name": "a",
      "annotation": "int"
    },
    {
      "name": "b",
      "annotation": "int"
    }
  ],
  "return_annotation": "int",
  "start_line": 1,
  "end_line": 2
}
```

### `chunker.py`

Transforms structured AST output into retrieval-friendly code units.

Instead of cutting source code every fixed number of characters, the chunker aims to preserve meaningful boundaries.

Example:

```text
operations.py
    |
    +-- add()       -> function chunk
    +-- subtract()  -> function chunk
    +-- multiply()  -> function chunk
    +-- divide()    -> function chunk
```

This improves retrieval because relevant functions or methods can be retrieved independently.

### `embedding_service.py`

Converts code chunks into vectors using Ollama.

```text
Code Chunk
    |
    v
Ollama Embedding Model
    |
    v
Vector
```

### `indexer.py`

Coordinates:

```text
AST
 |
 v
Chunking
 |
 v
Embeddings
 |
 v
ChromaDB
```

Indexing responses can include information such as:

```text
project_id
file_count
chunk_count
embedding_count
indexed_count
embedding_model
collection_name
status
```

### `retrieval_service.py`

Retrieves relevant project chunks for a question.

Useful retrieval metadata can include:

```text
project_id
file_path
chunk_kind
symbol_name
start_line
end_line
```

### `ollama_service.py`

Provides the shared interface to locally running Ollama models.

It can be reused for:

- Ask Code
- Code explanation
- File documentation
- Project summaries

### `rag_service.py`

Coordinates retrieval and generation:

```text
Question
   |
   v
Retrieve Context
   |
   v
Build Prompt
   |
   v
Ollama
   |
   v
Answer
```

### `doc_generator.py`

Generates project documentation from:

```text
AST metadata
+
Original source code
+
Ollama
+
ReportLab
```

ChromaDB is not required for the normal documentation-generation path.


---

## Installation

### Prerequisites

Install:

- Python 3.9+
- Git
- Ollama

Recommended:

- Python virtual environment
- A modern web browser

Clone the repository:

```bash
git clone (https://github.com/Bharathreddyshyamala/intelligent-code-doc-assistant)
cd intelligent-code-doc-assistant
```

---

## Backend Setup

Move into the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
fastapi
uvicorn
pydantic
requests
python-dotenv
chromadb
ollama
reportlab
python-multipart
```

Use the repository's `requirements.txt` as the source of truth for exact package versions.

---

## Ollama Setup

Verify Ollama:

```bash
ollama --version
```

The default Ollama server is:

```text
http://127.0.0.1:11434
```

Pull the embedding model:

```bash
ollama pull embeddinggemma
```

Pull the code-generation model:

```bash
ollama pull qwen2.5-coder
```

List installed models:

```bash
ollama list
```

---

## Environment Configuration

Create:

```text
backend/.env
```

Example:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=embeddinggemma
OLLAMA_CHAT_MODEL=qwen2.5-coder
```

If the code uses different environment-variable names, update the file to match the implementation.

A safe `.env.example` can contain:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=embeddinggemma
OLLAMA_CHAT_MODEL=qwen2.5-coder
```

Do not commit secrets or local-only configuration inside `.env`.

---

## Running the Application

### 1. Start Ollama

Make sure Ollama is running.

Optional check:

```bash
curl http://127.0.0.1:11434
```

### 2. Start FastAPI

From:

```text
intelligent-code-doc-assistant/backend
```

activate the environment:

```bash
source venv/bin/activate
```

Run:

```bash
uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 3. Start Streamlit

Open another terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```

Streamlit will print the local browser URL in the terminal.

---

## Using the Application

### Step 1 - Check Backend Health

Click:

```text
Check Backend Health
```

A successful response confirms that Streamlit can communicate with FastAPI.

### Step 2 - Ingest a Project

Choose one:

#### Local folder

Example:

```text
/Users/yourname/Desktop/calculator
```

#### GitHub repository

Example:

```text
https://github.com/example/calculator.git
```

#### ZIP upload

Example:

```text
calculator.zip
```

After ingestion, the UI can display:

- Project name
- Project ID
- Source type
- File count
- Project status

### Step 3 - Parse Source Code

Click:

```text
Parse Source Code
```

The backend creates structured AST output.

Primary result:

```text
ast.json
```

### Step 4 - Index Project

Click:

```text
Index Project
```

The application performs:

```text
Chunking
    |
    v
Embedding Generation
    |
    v
ChromaDB Indexing
```

The UI can show:

- Indexing status
- Files processed
- Chunks created
- Vectors indexed
- Embedding model
- ChromaDB collection

### Step 5 - Ask Code

After indexing, enter a question.

Example:

```text
Explain how the divide function handles division by zero.
```

The system retrieves relevant source chunks and sends grounded code context to Ollama.

### Step 6 - Generate PDF Documentation

After parsing, open **Generate Documentation**.

The detected project name is used as the default title.

Example:

```text
calculator
    ->
Calculator
```

Choose the maximum number of source files to include.

Example:

```text
30
```

This means **up to 30 files**, not exactly 30.

Examples:

```text
Project has 2 files
Maximum = 30
Result = 2 files documented
```

```text
Project has 100 files
Maximum = 30
Result = 30 files selected for that run
```

Click:

```text
Generate PDF Documentation
```

Then download the generated project PDF.

---

## API Overview

Always verify exact request schemas and route paths using:

```text
http://127.0.0.1:8000/docs
```

### Health

```http
GET /health
```

### Local project ingestion

```http
POST /ingest-local
```

Example:

```json
{
  "path": "/path/to/project"
}
```

### GitHub ingestion

```http
POST /ingest-github
```

Example:

```json
{
  "repo_url": "https://github.com/example/repository.git",
  "branch": "main"
}
```

### ZIP ingestion

```http
POST /ingest-upload
```

Uses multipart file upload.

### Parse project

```http
POST /parse-code
```

Example:

```json
{
  "project_id": "PROJECT_ID"
}
```

### Index project

```http
POST /index-code
```

Example:

```json
{
  "project_id": "PROJECT_ID"
}
```

### Ask Code

The Ask Code route accepts the current project ID and a code-related question.

Verify the exact route and request body in Swagger because its exact path depends on the current `rag_routes.py` implementation.

### Generate documentation

The current documentation frontend uses the versioned API prefix:

```http
POST /api/v1/generate-docs
```

Example:

```json
{
  "project_id": "PROJECT_ID",
  "project_title": "Calculator",
  "max_files": 30,
  "overwrite": true
}
```

### Preview generated PDF

```http
GET /api/v1/generated-docs/{project_id}
```

### Download generated PDF

```http
GET /api/v1/generated-docs/{project_id}/download
```

---

## Generated Project Data

Each project is associated with its unique `project_id`.

Conceptually:

```text
Project
   |
   +-- Source Files
   |
   +-- Project Metadata
   |
   +-- ast.json
   |
   +-- chunks.json
   |
   +-- ChromaDB Records
   |
   +-- Generated Documentation
```

### `ast.json`

Stores structured static-analysis output.

### `chunks.json`

Stores retrieval-ready code chunks and metadata.

### ChromaDB

Stores:

```text
Embedding
+
Chunk Content
+
Metadata
```

### `generated_docs/`

Example:

```text
backend/generated_docs/
|
+-- <project_id>/
    |
    +-- PROJECT_DOCUMENTATION.pdf
    +-- manifest.json
```

The Streamlit download filename can be project specific:

```text
Calculator_Documentation.pdf
```

---

## RAG Pipeline

RAG stands for:

```text
Retrieval-Augmented Generation
```

Instead of asking the LLM to answer only from general model knowledge, the application retrieves code from the indexed project and includes that code in the prompt.

```text
User Question
      |
      v
Question Embedding
      |
      v
Vector Retrieval
      |
      v
Relevant Code Chunks
      |
      v
Context Construction
      |
      v
Ollama
      |
      v
Grounded Answer
```

This is especially useful for repository-specific questions.

Example:

```text
Where is indexing performed?
```

Potential relevant files:

```text
routes/indexing_routes.py
services/indexer.py
services/chunker.py
services/embedding_service.py
```

The LLM can then explain the flow using actual project code.

---

## Documentation Generation

The current documentation generator intentionally uses:

```text
AST + Source Code
```

rather than ChromaDB as its primary source.

### Why AST is used

AST gives deterministic project structure:

- Which files exist
- Which classes exist
- Which functions exist
- Parameters
- Docstrings
- Imports
- Line ranges
- Calls
- Exceptions
- Control flow

### Why ChromaDB is not the primary source for docs

ChromaDB answers:

```text
Which chunks are most relevant to this query?
```

Documentation generation needs:

```text
Systematically document the selected project files.
```

Similarity search could miss code that is important but not highly similar to a specific query.

Therefore:

```text
Ask Code
    -> ChromaDB Retrieval

Generate Docs
    -> AST + Source Code
```

### Maximum source files

The documentation UI allows the user to control the number of selected files.

Examples:

| Project Files | Maximum | Documented |
|---:|---:|---:|
| 2 | 30 | 2 |
| 20 | 30 | 20 |
| 30 | 30 | 30 |
| 100 | 30 | 30 |

The limit exists because the current design can make approximately one Ollama request per selected file plus one project-summary request.

### Regenerate existing documentation

When:

```text
overwrite = true
```

existing generated documentation for the project is replaced.

This does **not** modify:

- Source files
- `ast.json`
- `chunks.json`
- ChromaDB vectors

It only affects generated documentation files.

---

## Example Workflow

Consider:

```text
calculator/
|
+-- main.py
+-- operations.py
```

Example source:

```python
def add(a, b):
    return a + b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")

    return a / b
```

### AST parsing

The parser can identify:

```text
Function: add
Parameters: a, b
Return: a + b

Function: divide
Parameters: a, b
Condition: b == 0
Raised exception: ValueError
Return: a / b
```

### Chunking

The chunker can create:

```text
Chunk 1 -> add()
Chunk 2 -> divide()
```

### Indexing

Each chunk is embedded and stored in ChromaDB.

### Ask Code

Question:

```text
How does division by zero work?
```

The `divide()` chunk is retrieved and sent to Ollama as context.

### Generate Docs

The documentation generator uses:

```text
Calculator AST
+
Calculator Source Code
```

and generates:

```text
Calculator_Documentation.pdf
```

---

## Local-First Design

The application is designed for local execution.

With local Ollama and ChromaDB:

- Source code stays on the local machine.
- Embeddings are generated locally.
- LLM inference runs locally.
- Vector data is stored locally.
- Generated PDFs are stored locally.

This is useful for:

- Private repositories
- Internal company projects
- Academic projects
- Offline workflows
- Source code that should not be sent to cloud LLM APIs

---

## Troubleshooting

### Port 8000 already in use

Check:

```bash
lsof -i :8000
```

Stop the conflicting process or use another port:

```bash
uvicorn main:app --reload --port 8001
```

If the port changes, update the frontend backend URL.

### Could not import module `main`

Run Uvicorn from the folder containing `main.py`.

```bash
cd backend
uvicorn main:app --reload
```

### Streamlit command not found

Install Streamlit in the active environment:

```bash
pip install streamlit
```

Then:

```bash
streamlit run streamlit_app.py
```

### Ollama connection failure

Check:

```bash
ollama --version
ollama list
```

Confirm Ollama is available at:

```text
http://127.0.0.1:11434
```

### Missing embedding model

```bash
ollama pull embeddinggemma
```

### Missing code model

```bash
ollama pull qwen2.5-coder
```

### Generate Docs returns 404

Open:

```text
http://127.0.0.1:8000/docs
```

Verify that the documentation route is registered.

The current frontend expects:

```text
/api/v1/generate-docs
```

### PDF generation fails

Install ReportLab:

```bash
pip install reportlab
```

### Ask Code does not work after parsing

Ask Code requires indexing.

Use:

```text
Ingest
 -> Parse
 -> Index
 -> Ask Code
```

### Generate Docs without indexing

The current documentation flow can use:

```text
Ingest
 -> Parse
 -> Generate Documentation
```

because it reads `ast.json` and source files directly.

---

## Current Limitations

- Deep AST analysis is currently focused on Python.
- Large repositories can take significant time to process locally.
- Documentation generation can require many Ollama calls.
- The current maximum-file option selects only the first selected subset for that run.
- Repeated runs with the same maximum do not automatically continue with the next batch unless batch-offset logic is implemented.
- LLM speed depends on local CPU/GPU/RAM.
- Generated documentation should still be reviewed for accuracy.
- Cross-file architecture explanations depend on the available parsed and retrieved context.

---

## Future Enhancements

Potential next improvements include:

- Hybrid retrieval using dense + sparse search
- Cross-encoder reranking
- LangGraph agent routing
- Dedicated Explain Code mode
- Multi-language parsing
- Large-repository batch documentation
- Automatic continuation across documentation batches
- Merge multiple documentation batches into one final PDF
- Incremental documentation regeneration
- Changed-file-only documentation
- Dependency graph visualization
- Call graph visualization
- Architecture diagrams
- RAGAS evaluation
- Retrieval metrics
- Faithfulness evaluation
- Source citations in Ask Code
- Git-diff-aware re-indexing
- Streaming Ollama responses
- Background processing
- Progress reporting for large repositories

Only mark a feature as implemented in this README after the corresponding code has been completed and tested.

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
venv/
.venv/

# Environment
.env

# IDE
.vscode/
.idea/

# macOS
.DS_Store

# Frontend / JavaScript
node_modules/
.next/

# Runtime project data
backend/temp_repos/
backend/uploads/
backend/generated_docs/
backend/vector_store/

# Local vector stores
chroma_db/
```

Keep `.env.example` committed, but keep `.env` private.

---

## Contributing

A typical Git workflow:

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature
```

After changes:

```bash
git add .
git commit -m "Add feature description"
git push origin feature/your-feature
```

Then open a pull request.

Keep responsibilities separated between:

```text
Routes
Services
Parsing
Chunking
Embedding
Indexing
Retrieval
Generation
Frontend
```

---

## Project Status

The implemented core pipeline is:

```text
Project Ingestion
      |
      v
AST Parsing
      |
      v
Code Chunking
      |
      v
Ollama Embeddings
      |
      v
ChromaDB Indexing
      |
      v
Ask Code
```

The documentation pipeline is:

```text
Project Ingestion
      |
      v
AST Parsing
      |
      v
AST + Source Code
      |
      v
Ollama Documentation Generation
      |
      v
PDF Documentation
```

Together, these provide the foundation for a local intelligent code-understanding and documentation assistant.

---

