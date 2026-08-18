"""Thin wrapper over the local Ollama chat endpoint."""

from __future__ import annotations

import json
from typing import Any, Iterator

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


def chat_stream(messages: list[dict[str, Any]], temperature: float = 0.2) -> Iterator[str]:
    """Yield answer fragments as Ollama produces them.

    Ollama streams newline-delimited JSON objects; each carries the next piece
    of the message in message.content, and the final one sets done=true.
    """
    url = f"{settings.ollama_host}/api/chat"
    try:
        with requests.post(
            url,
            json={
                "model": settings.llm_model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
            timeout=TIMEOUT,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fragment = payload.get("message", {}).get("content", "")
                if fragment:
                    yield fragment
                if payload.get("done"):
                    return
    except requests.RequestException as exc:
        raise LLMError(
            f"LLM stream from {url} failed ({exc}). "
            f"Is Ollama running and '{settings.llm_model}' pulled?"
        ) from exc


def ollama_reachable() -> bool:
    try:
        requests.get(f"{settings.ollama_host}/api/tags", timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False
