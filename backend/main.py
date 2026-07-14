from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime


app = FastAPI(
    title="Intelligent Code Documentation Assistant API",
    description="FastAPI backend for Local RAG and Agentic AI code documentation assistant",
    version="1.0.0"
)


# CORS setup allows Streamlit frontend to call FastAPI backend
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Intelligent Code Documentation Assistant API",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running successfully",
        "service": "Intelligent Code Documentation Assistant",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }