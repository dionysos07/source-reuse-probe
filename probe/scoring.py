"""Scoring rule v1: content-word n-gram overlap between an answer and a source.

The rule's parameters live in config/scoring.yaml, never here, so that changing
the rule produces a diff in the config rather than a silent change in behaviour.
This module implements exactly that one rule; there is no plug-in mechanism.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

# Letters and digits, optionally joined by an apostrophe, so that "don't" stays
# one token and can be matched against the stopword list as written.
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

_REQUIRED_KEYS = (
    "version",
    "ngram_size",
    "casefold",
    "strip_diacritics",
    "stopwords_file",
    "reuse_threshold",
)


class ConfigError(Exception):
    """Raised when scoring.yaml is missing or malformed, so the CLI can say so."""


@dataclass(frozen=True)
class ScoringConfig:
    version: str
    ngram_size: int
    stopwords: frozenset[str]
    casefold: bool
    strip_diacritics: bool
    reuse_threshold: float

    @classmethod
    def load(cls, path: Path) -> "ScoringConfig":
        """Read scoring.yaml. `stopwords_file` is resolved next to that file."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise ConfigError(f"scoring config not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"scoring config {path} is not valid YAML: {exc}") from exc

        missing = [key for key in _REQUIRED_KEYS if key not in raw]
        if missing:
            raise ConfigError(f"scoring config {path} is missing: {', '.join(missing)}")

        if not isinstance(raw["ngram_size"], int) or raw["ngram_size"] < 1:
            raise ConfigError(f"ngram_size must be a positive integer, got {raw['ngram_size']!r}")

        casefold = bool(raw["casefold"])
        stopwords_path = path.parent / str(raw["stopwords_file"])
        try:
            lines = stopwords_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise ConfigError(f"stopwords file not found: {stopwords_path}") from exc

        words = (line.strip() for line in lines)
        words = (word for word in words if word and not word.startswith("#"))
        stopwords = frozenset(word.casefold() if casefold else word for word in words)

        return cls(
            version=str(raw["version"]),
            ngram_size=int(raw["ngram_size"]),
            stopwords=stopwords,
            casefold=casefold,
            strip_diacritics=bool(raw["strip_diacritics"]),
            reuse_threshold=float(raw["reuse_threshold"]),
        )


@dataclass(frozen=True)
class OverlapScore:
    """A score plus the counts behind it.

    `matched`/`total` are kept because 0.0 from "nothing in common" and 0.0 from
    "the answer was too short to form a single n-gram" are different facts, and
    the second must not be averaged in as if it were evidence of no reuse.
    """

    score: float
    matched: int
    total: int
    config_version: str

    @property
    def scorable(self) -> bool:
        return self.total > 0


def normalise(text: str, config: ScoringConfig) -> list[str]:
    """Reduce text to its content words, in order.

    Separate from scoring so the tests can pin tokenisation on its own, and so a
    surprising score can be explained by printing what the scorer actually saw.
    """
    text = unicodedata.normalize("NFC", text).replace("’", "'")
    if config.casefold:
        text = text.casefold()
    if config.strip_diacritics:
        text = _strip_diacritics(text)

    return [token for token in _TOKEN.findall(text) if token not in config.stopwords]


def score_overlap(answer: str, source_text: str, config: ScoringConfig) -> OverlapScore:
    """Fraction of the answer's content-word n-grams that occur in the source.

    Answer n-grams are counted with multiplicity: a phrase the answer repeats
    three times is three chances to match, which is the literal reading of
    "fraction of the answer's n-grams" and keeps the measure positional.
    """
    answer_ngrams = _ngrams(normalise(answer, config), config.ngram_size)
    if not answer_ngrams:
        return OverlapScore(0.0, 0, 0, config.version)

    source_ngrams = set(_ngrams(normalise(source_text, config), config.ngram_size))
    matched = sum(1 for ngram in answer_ngrams if ngram in source_ngrams)

    return OverlapScore(
        score=matched / len(answer_ngrams),
        matched=matched,
        total=len(answer_ngrams),
        config_version=config.version,
    )


def _ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )
