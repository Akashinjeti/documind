from __future__ import annotations

import json
import re
from typing import Protocol

import httpx


class LLM(Protocol):
    def complete(self, role: str, system: str, user: str) -> str:
        """`role` names the calling agent: planner | answerer | critic."""
        ...


class GroqLLM:
    """Groq's OpenAI-compatible chat-completions endpoint (LLaMA 3.3 70B by default)."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 60.0) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self._model = model

    def complete(self, role: str, system: str, user: str) -> str:
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class DemoLLM:
    """Keyless fallback so the whole pipeline runs offline.

    Planner passes the question through, the answerer quotes the best context
    passage (extractive), the critic approves. Set GROQ_API_KEY for real answers.
    """

    def complete(self, role: str, system: str, user: str) -> str:
        if role == "planner":
            q = re.search(r"Question: (.*)", user)
            return json.dumps({"queries": [q.group(1).strip() if q else user]})
        if role == "answerer":
            passages = re.findall(r"\[(\d+)\] (.*?)(?=\n\[\d+\] |\Z)", user, flags=re.S)
            if not passages:
                return "I could not find this in the ingested documents."
            n, text = passages[0]
            return f"{text.strip()} [{n}]"
        if role == "critic":
            return json.dumps({"grounded": True, "issues": "", "better_query": ""})
        raise ValueError(f"unknown role {role!r}")


def parse_json(raw: str) -> dict:
    """Tolerant JSON extraction: handles code fences and prose around the object."""
    raw = raw.strip()
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                out = json.loads(m.group(0))
                return out if isinstance(out, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}
