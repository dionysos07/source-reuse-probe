"""The two model stages: inventing questions, then answering them blind.

Prompts are module constants rather than config because their exact text is
recorded in every result; keeping them in code makes a prompt change a reviewable
diff next to the code that sends it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from probe.fetch import Article
from probe.model import ModelConfig, ModelError, ModelReply, complete
from probe.records import read_records, write_records

QUESTION_SYSTEM = (
    "You write questions that a member of the public might type into a search "
    "engine. You reply with questions only, one per line, no numbering and no "
    "commentary."
)

QUESTION_TEMPLATE = """Below is a health article. Write {count} questions that this article answers.

Rules:
- Each question must be answerable from the article.
- Write them as a member of the public would ask them, in general terms.
- Do not refer to "the article", "the text", or "this page".
- One question per line, no numbering.

ARTICLE:
{article}"""

# Deliberately says nothing about the source site, and offers no retrieval: the
# whole point is to see what the model reproduces unprompted.
ANSWER_SYSTEM = (
    "You answer general health questions for a member of the public. Answer "
    "directly and concisely from your own knowledge. Do not ask clarifying "
    "questions and do not tell the user to consult a website."
)

ANSWER_TEMPLATE = "{question}"

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


@dataclass(frozen=True)
class QuestionRecord:
    id: str
    article_url: str
    question: str
    system_prompt: str
    user_prompt: str
    model_requested: str
    model_served: str
    created_at: str


@dataclass(frozen=True)
class AnswerRecord:
    question_id: str
    article_url: str
    question: str
    answer: str
    system_prompt: str
    user_prompt: str
    model_requested: str
    model_served: str
    created_at: str


def generate_questions(
    articles: list[Article], config: ModelConfig, path: Path
) -> list[QuestionRecord]:
    """Ask the model for questions each article answers, and store them."""
    records: list[QuestionRecord] = []

    for index, article in enumerate(articles):
        prompt = QUESTION_TEMPLATE.format(count=config.questions_per_article, article=article.text)
        reply = complete(QUESTION_SYSTEM, prompt, config)

        questions = _parse_questions(reply.text, config.questions_per_article)
        if not questions:
            raise ModelError(f"no questions parsed from the reply for {article.url}")

        for position, question in enumerate(questions):
            records.append(
                QuestionRecord(
                    id=f"q{index:03d}-{position}",
                    article_url=article.url,
                    question=question,
                    system_prompt=reply.system_prompt,
                    user_prompt=reply.user_prompt,
                    model_requested=reply.model_requested,
                    model_served=reply.model_served,
                    created_at=_now(),
                )
            )

    write_records(path, [asdict(record) for record in records])
    return records


def answer_questions(
    questions: list[QuestionRecord], config: ModelConfig, path: Path
) -> list[AnswerRecord]:
    """Put each question to the model with no retrieval and no source named."""
    records: list[AnswerRecord] = []

    for question in questions:
        reply: ModelReply = complete(
            ANSWER_SYSTEM, ANSWER_TEMPLATE.format(question=question.question), config
        )
        records.append(
            AnswerRecord(
                question_id=question.id,
                article_url=question.article_url,
                question=question.question,
                answer=reply.text,
                system_prompt=reply.system_prompt,
                user_prompt=reply.user_prompt,
                model_requested=reply.model_requested,
                model_served=reply.model_served,
                created_at=_now(),
            )
        )

    write_records(path, [asdict(record) for record in records])
    return records


def load_questions(path: Path) -> list[QuestionRecord]:
    return [QuestionRecord(**row) for row in read_records(path, "questions")]


def load_answers(path: Path) -> list[AnswerRecord]:
    return [AnswerRecord(**row) for row in read_records(path, "answers")]


def _parse_questions(text: str, limit: int) -> list[str]:
    """Take one question per line, tolerating numbering the model added anyway."""
    lines = (_LIST_MARKER.sub("", line).strip() for line in text.splitlines())
    return [line for line in lines if line][:limit]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
