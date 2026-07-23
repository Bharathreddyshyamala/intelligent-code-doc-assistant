import os
from typing import Any, List, Sequence

import ollama
from dotenv import load_dotenv


load_dotenv()


# ---------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "embeddinggemma",
)

OLLAMA_CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "qwen2.5-coder:3b",
)

OLLAMA_CHAT_TEMPERATURE = float(
    os.getenv(
        "OLLAMA_CHAT_TEMPERATURE",
        "0.2",
    )
)


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------


class OllamaServiceError(RuntimeError):
    """
    Raised when communication with Ollama fails.
    """


# ---------------------------------------------------------
# Ollama client
# ---------------------------------------------------------


def get_ollama_client() -> ollama.Client:
    """
    Create a client connected to the configured Ollama server.
    """

    return ollama.Client(
        host=OLLAMA_BASE_URL,
    )


# ---------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------


def extract_embeddings(
    response: Any,
) -> List[List[float]]:
    """
    Extract embedding vectors from an Ollama response.

    The Ollama Python package may return either:
    - an object-style response
    - a dictionary-style response
    """

    embeddings = getattr(
        response,
        "embeddings",
        None,
    )

    if embeddings is None:
        try:
            embeddings = response["embeddings"]

        except (KeyError, TypeError) as exc:
            raise OllamaServiceError(
                "Ollama response did not contain embeddings."
            ) from exc

    if embeddings is None:
        raise OllamaServiceError(
            "Ollama returned an empty embeddings response."
        )

    return [
        list(embedding)
        for embedding in embeddings
    ]


def generate_embeddings(
    texts: Sequence[str],
    model: str = OLLAMA_EMBED_MODEL,
) -> List[List[float]]:
    """
    Generate one embedding vector for every supplied text.
    """

    if not texts:
        return []

    input_texts = list(texts)

    try:
        client = get_ollama_client()

        response = client.embed(
            model=model,
            input=input_texts,
        )

    except Exception as exc:
        raise OllamaServiceError(
            "Unable to communicate with Ollama while "
            "generating embeddings. "
            f"Server: '{OLLAMA_BASE_URL}'. "
            f"Model: '{model}'. "
            "Make sure Ollama is running and the model is "
            f"installed. Original error: {exc}"
        ) from exc

    embeddings = extract_embeddings(
        response
    )

    if len(embeddings) != len(input_texts):
        raise OllamaServiceError(
            "The number of embeddings returned by Ollama "
            "does not match the number of input texts. "
            f"Inputs: {len(input_texts)}, "
            f"embeddings: {len(embeddings)}."
        )

    return embeddings


# ---------------------------------------------------------
# Chat response helpers
# ---------------------------------------------------------


def extract_chat_content(
    response: Any,
) -> str:
    """
    Extract assistant text from an Ollama chat response.
    """

    message = getattr(
        response,
        "message",
        None,
    )

    if message is None:
        try:
            message = response["message"]

        except (KeyError, TypeError) as exc:
            raise OllamaServiceError(
                "Ollama chat response did not contain "
                "a message."
            ) from exc

    content = getattr(
        message,
        "content",
        None,
    )

    if content is None and isinstance(message, dict):
        content = message.get("content")

    if not content or not content.strip():
        raise OllamaServiceError(
            "Ollama returned an empty chat response."
        )

    return content.strip()


def generate_chat_response(
    prompt: str,
    system_prompt: str,
    model: str = OLLAMA_CHAT_MODEL,
    temperature: float = OLLAMA_CHAT_TEMPERATURE,
) -> str:
    """
    Generate a natural-language answer using an Ollama
    chat model.
    """

    cleaned_prompt = prompt.strip()
    cleaned_system_prompt = system_prompt.strip()

    if not cleaned_prompt:
        raise ValueError(
            "The chat prompt cannot be empty."
        )

    if not cleaned_system_prompt:
        raise ValueError(
            "The system prompt cannot be empty."
        )

    try:
        client = get_ollama_client()

        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": cleaned_system_prompt,
                },
                {
                    "role": "user",
                    "content": cleaned_prompt,
                },
            ],
            stream=False,
            options={
                "temperature": temperature,
            },
        )

    except Exception as exc:
        raise OllamaServiceError(
            "Unable to generate a chat response using "
            f"Ollama model '{model}'. "
            "Make sure Ollama is running and the chat model "
            f"is installed. Original error: {exc}"
        ) from exc

    return extract_chat_content(
        response
    )