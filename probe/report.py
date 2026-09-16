"""Score every stored answer against its source article and report the result.

This is the stage that must be rerunnable for free: scoring touches no network
and no model, so a change to config/scoring.yaml can be re-measured immediately.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from probe.fetch import Article
from probe.generate import AnswerRecord
from probe.records import write_records
from probe.scoring import ScoringConfig, score_overlap


@dataclass(frozen=True)
class Result:
    """One scored (answer, source article) pair.

    Carries everything needed to reproduce itself: both model ids, the exact
    prompts, the scoring config version and when it was scored.
    """

    question_id: str
    article_url: str
    article_title: str
    article_sha256: str
    question: str
    answer: str
    score: float
    matched_ngrams: int
    total_ngrams: int
    scorable: bool
    scoring_version: str
    model_requested: str
    model_served: str
    answer_system_prompt: str
    answer_user_prompt: str
    answered_at: str
    scored_at: str


def score_answers(
    answers: list[AnswerRecord], articles: list[Article], config: ScoringConfig
) -> list[Result]:
    by_url = {article.url: article for article in articles}
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[Result] = []

    for answer in answers:
        article = by_url.get(answer.article_url)
        if article is None:
            # The answer's source is not in the cache, so there is nothing to
            # score it against. Skipping beats inventing a zero.
            continue

        overlap = score_overlap(answer.answer, article.text, config)
        results.append(
            Result(
                question_id=answer.question_id,
                article_url=answer.article_url,
                article_title=article.title,
                article_sha256=article.text_sha256,
                question=answer.question,
                answer=answer.answer,
                score=round(overlap.score, 6),
                matched_ngrams=overlap.matched,
                total_ngrams=overlap.total,
                scorable=overlap.scorable,
                scoring_version=overlap.config_version,
                model_requested=answer.model_requested,
                model_served=answer.model_served,
                answer_system_prompt=answer.system_prompt,
                answer_user_prompt=answer.user_prompt,
                answered_at=answer.created_at,
                scored_at=scored_at,
            )
        )

    return results


def write_results(path: Path, results: list[Result]) -> None:
    write_records(path, [asdict(result) for result in results])


def format_summary(results: list[Result], config: ScoringConfig) -> str:
    """A per-article table plus totals, as plain text for a terminal."""
    if not results:
        return "No results to report."

    lines = [
        f"scoring rule {config.version}: {config.ngram_size}-gram content-word overlap",
        f"reuse threshold {config.reuse_threshold:g}",
        "",
        f"{'article':<44} {'n':>3} {'mean':>6} {'max':>6} {'>=thr':>6}",
        "-" * 68,
    ]

    for url in dict.fromkeys(result.article_url for result in results):
        rows = [result for result in results if result.article_url == url]
        lines.append(_row(_label(rows[0]), rows, config))

    lines.append("-" * 68)
    lines.append(_row("ALL", results, config))

    unscorable = [result for result in results if not result.scorable]
    if unscorable:
        lines.append("")
        lines.append(
            f"{len(unscorable)} answer(s) too short to form a "
            f"{config.ngram_size}-gram, excluded from the means above"
        )

    return "\n".join(lines)


def _row(label: str, rows: list[Result], config: ScoringConfig) -> str:
    scorable = [row.score for row in rows if row.scorable]
    over = sum(1 for score in scorable if score >= config.reuse_threshold)
    if not scorable:
        return f"{label:<44} {len(rows):>3} {'-':>6} {'-':>6} {'-':>6}"
    return (
        f"{label:<44} {len(rows):>3} {statistics.mean(scorable):>6.3f} "
        f"{max(scorable):>6.3f} {over:>6}"
    )


def _label(result: Result) -> str:
    label = result.article_title or result.article_url
    return label if len(label) <= 44 else label[:41] + "..."
