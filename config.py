"""
config.py

Centralised, environment-driven configuration for the LLM layer.

"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for talking to a local Ollama server."""

    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "60"))
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", "memory.db")
