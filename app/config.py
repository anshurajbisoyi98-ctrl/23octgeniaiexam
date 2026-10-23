"""Project settings, read from environment variables or the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The repository root (the folder that contains app/, data/, frontend/ ...)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Files and folders
    pdf_path: Path = ROOT_DIR / "data" / "GER-3620-P.pdf"
    chroma_dir: Path = ROOT_DIR / "chroma_db"
    collection_name: str = "ger3620"
    frontend_dir: Path = ROOT_DIR / "frontend"

    # Embeddings (local model, no API key needed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 5
    # Cosine similarity (0..1). Below this, we answer "Not found" without calling the LLM.
    relevance_threshold: float = 0.65

    # LLM
    llm_provider: str = "gemini"  # gemini | groq | ollama
    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"


@lru_cache
def get_settings() -> Settings:
    return Settings()
