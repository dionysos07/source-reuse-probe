"""The single model call that stands in for a search engine's answer generation.

One provider, called directly. There is deliberately no provider abstraction:
swapping providers should be a visible edit to this file, not a config change
that quietly alters what the numbers mean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
import yaml

from google import genai
from google.genai import types, errors

client = genai.Client()                      # reads GEMINI_API_KEY
response = client.models.generate_content(
    model=config.model,
    contents=user_prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=config.max_tokens,
    ),
)

_REQUIRED_KEYS = ("model", "max_tokens", "questions_per_article")


class ModelError(Exception):
    """Any failure talking to the model, reported to the user as a message."""


@dataclass(frozen=True)
class ModelConfig:
    model: str
    max_tokens: int
    questions_per_article: int

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise ModelError(f"model config not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ModelError(f"model config {path} is not valid YAML: {exc}") from exc

        missing = [key for key in _REQUIRED_KEYS if key not in raw]
        if missing:
            raise ModelError(f"model config {path} is missing: {', '.join(missing)}")

        return cls(
            model=str(raw["model"]),
            max_tokens=int(raw["max_tokens"]),
            questions_per_article=int(raw["questions_per_article"]),
        )


@dataclass(frozen=True)
class ModelReply:
    text: str
    model_requested: str
    model_served: str
    system_prompt: str
    user_prompt: str


def complete(system_prompt: str, user_prompt: str, config: ModelConfig) -> ModelReply:
    """Send one prompt and return the reply with both model ids.

    `model_served` is the id the API reports back, which can differ from the id
    we asked for when an alias resolves to a newer snapshot. Recording only the
    requested id would make a result impossible to place in time.
    """
    client = _client()

    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise ModelError("ANTHROPIC_API_KEY was rejected by the API") from exc
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "unknown")
        raise ModelError(f"rate limited by the API; retry after {retry_after}s") from exc
    except anthropic.APIStatusError as exc:
        raise ModelError(f"API returned {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ModelError(f"could not reach the API: {exc}") from exc

    if response.stop_reason == "refusal":
        raise ModelError(f"the model declined to answer: {user_prompt[:80]!r}")

    text = "\n".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ModelError(f"the model returned no text for: {user_prompt[:80]!r}")

    return ModelReply(
        text=text,
        model_requested=config.model,
        model_served=response.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ModelError(
            "ANTHROPIC_API_KEY is not set. Export it before running the "
            "questions or answer stage; it is never read from a file in this repo."
        )
    return anthropic.Anthropic()
