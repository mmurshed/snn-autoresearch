"""LLM backends for surrogate gradient generation."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

from .candidate import SurrogateCandidate


class LLM(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


class ClaudeLLM(LLM):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model

    def generate(self, system: str, user: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.7,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  LLM retry in {wait}s: {e}")
                time.sleep(wait)


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def generate(self, system: str, user: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  LLM retry in {wait}s: {e}")
                time.sleep(wait)


def get_llm(backend: str = "claude") -> LLM:
    """Create an LLM backend. Requires ANTHROPIC_API_KEY or OPENAI_API_KEY."""
    if backend == "claude":
        return ClaudeLLM()
    elif backend == "openai":
        return OpenAILLM()
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'claude' or 'openai'.")


def parse_candidates(text: str) -> list[SurrogateCandidate]:
    """Extract SurrogateCandidate objects from LLM response text."""
    candidates = []
    # Match JSON objects containing required fields
    pattern = r"\{[^{}]*\"name\"[^{}]*\"python_expr\"[^{}]*\}"
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            raw = match.group()
            raw = raw.replace("\u2018", '"').replace("\u2019", '"')
            raw = raw.replace("\u201c", '"').replace("\u201d", '"')
            raw = re.sub(r",\s*}", "}", raw)
            data = json.loads(raw)
            candidates.append(
                SurrogateCandidate(
                    name=data["name"],
                    symbolic_expr=data.get("symbolic_expr", ""),
                    python_expr=data["python_expr"],
                    params=data.get("params", {}),
                    source="llm",
                    generation=0,
                    reasoning=data.get("reasoning", ""),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return candidates
