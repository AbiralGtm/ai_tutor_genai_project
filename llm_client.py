from __future__ import annotations

from typing import Optional

import requests

from config import Settings


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server cannot be reached at all."""


class OllamaResponseError(RuntimeError):
    """Raised when Ollama responds but reports an error (e.g. unknown model)."""


class OllamaClient:
    """
    Minimal client for Ollama's local /api/chat endpoint.

    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()

    def health_check(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            response = requests.get(
                f"{self.settings.host}/api/tags", timeout=self.settings.timeout
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def chat(self, messages: list[dict]) -> str:
        """
        Send a conversation to the model and return the full reply text.

        Each message is a dict, e.g. {"role": "user", "content": "..."}.

        Raises:
            OllamaConnectionError: the server could not be reached, or timed out.
            OllamaResponseError: the server responded with an error payload.
        """
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.settings.host}/api/chat",
                json=payload,
                timeout=self.settings.timeout,
            )
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.settings.host}. "
                "Is it running? Try: `ollama serve`"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaConnectionError(
                f"Timed out waiting for Ollama at {self.settings.host}."
            ) from exc

        if response.status_code != 200:
            raise OllamaResponseError(self._extract_error(response))

        data = response.json()

        error = data.get("error")
        if error:
            raise OllamaResponseError(error)

        return data.get("message", {}).get("content", "")

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Like chat(), but for Ollama's native tool-calling: passes `tools`
        (a list of JSON-schema function descriptions) and returns the full
        assistant message dict -- e.g. {"role": "assistant", "content": "",
        "tool_calls": [...]} -- instead of just the text content, so the
        caller can see and execute any tool calls the model requests.

        Raises:
            OllamaConnectionError: the server could not be reached, or timed out.
            OllamaResponseError: the server responded with an error payload.
        """
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.settings.host}/api/chat",
                json=payload,
                timeout=self.settings.timeout,
            )
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.settings.host}. "
                "Is it running? Try: `ollama serve`"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaConnectionError(
                f"Timed out waiting for Ollama at {self.settings.host}."
            ) from exc

        if response.status_code != 200:
            raise OllamaResponseError(self._extract_error(response))

        data = response.json()

        error = data.get("error")
        if error:
            raise OllamaResponseError(error)

        return data.get("message") or {"role": "assistant", "content": ""}

    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """
        Return the embedding vector for `text` from a local Ollama embedding
        model (defaults to settings.embedding_model, e.g. nomic-embed-text).

        Raises:
            OllamaConnectionError: the server could not be reached, or timed out.
            OllamaResponseError: the server responded with an error payload,
                or an unexpected/empty embedding.
        """
        payload = {
            "model": model or self.settings.embedding_model,
            "prompt": text,
        }

        try:
            response = requests.post(
                f"{self.settings.host}/api/embeddings",
                json=payload,
                timeout=self.settings.timeout,
            )
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.settings.host}. "
                "Is it running? Try: `ollama serve`"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaConnectionError(
                f"Timed out waiting for Ollama at {self.settings.host}."
            ) from exc

        if response.status_code != 200:
            raise OllamaResponseError(self._extract_error(response))

        data = response.json()

        error = data.get("error")
        if error:
            raise OllamaResponseError(error)

        embedding = data.get("embedding")
        if not embedding:
            raise OllamaResponseError("Ollama returned an empty embedding.")

        return embedding

    @staticmethod
    def _extract_error(response: requests.Response) -> str:
        try:
            return response.json().get("error", response.text)
        except ValueError:
            return response.text