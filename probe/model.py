"""The single model call that stands in for a search engine's answer generation.

One provider, called directly. There is deliberately no provider abstraction:
swapping providers should be a visible edit to this file, not a config change
that quietly alters what the numbers mean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from google import genai
from google.genai import errors, types

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
        response = client.models.generate_content(
            model=config.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=config.max_tokens,
                temperature=0.0,
            ),
        )
    except errors.ClientError as exc:
        if exc.code == 429:
            raise ModelError("rate limited by the API; wait and retry") from exc
        if exc.code in (401, 403):
            raise ModelError("GEMINI_API_KEY was rejected by the API") from exc
        raise ModelError(f"API returned {exc.code}: {exc.message}") from exc
    except errors.ServerError as exc:
        raise ModelError(f"API returned {exc.code}: {exc.message}") from exc
    except errors.APIError as exc:
        raise ModelError(f"could not reach the API: {exc}") from exc

    blocked = getattr(response.prompt_feedback, "block_reason", None)
    if blocked:
        raise ModelError(f"the prompt was blocked ({blocked}): {user_prompt[:80]!r}")

    if not response.candidates:
        raise ModelError(f"the API returned no candidate for: {user_prompt[:80]!r}")

    finish = response.candidates[0].finish_reason
    if finish not in (types.FinishReason.STOP, types.FinishReason.MAX_TOKENS):
        raise ModelError(f"the answer was filtered ({finish}): {user_prompt[:80]!r}")

    text = (response.text or "").strip()
    if not text:
        raise ModelError(f"the model returned no text for: {user_prompt[:80]!r}")

    return ModelReply(
        text=text,
        model_requested=config.model,
        model_served=response.model_version or config.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _client() -> genai.Client:
    if not os.environ.get("GEMINI_API_KEY"):
        raise ModelError(
            "GEMINI_API_KEY is not set. Set it before running the questions or "
            "answer stage; it is never read from a file in this repo."
        )
    return genai.Client()
