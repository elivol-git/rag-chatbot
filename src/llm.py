"""Thin wrapper over the local Ollama chat endpoint."""

from __future__ import annotations

from typing import Any

import requests

from .config import settings

TIMEOUT = 300


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
    url = f"{settings.ollama_host}/api/chat"
    try:
        response = requests.post(
            url,
            json={
                "model": settings.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(
            f"LLM request to {url} failed ({exc}). "
            f"Is Ollama running and '{settings.llm_model}' pulled?"
        ) from exc

    return response.json().get("message", {}).get("content", "").strip()


def ollama_reachable() -> bool:
    try:
        requests.get(f"{settings.ollama_host}/api/tags", timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False
