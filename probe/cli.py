"""Command line entry point.

The pipeline is split into separate subcommands so that re-scoring never costs
another model call, and a failed stage can be retried without repeating the ones
before it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from probe.extract import ExtractionError
from probe.fetch import ArticleCache, FetchError, SourceConfig, fetch_articles, read_url_list
from probe.generate import answer_questions, generate_questions, load_answers, load_questions
from probe.model import ModelConfig, ModelError
from probe.records import RecordError
from probe.report import format_summary, score_answers, write_results
from probe.scoring import ConfigError, ScoringConfig

# Paths are resolved relative to the repository, never from an absolute path or
# the caller's working directory.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_CONFIG = ROOT / "config"


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except (FetchError, ExtractionError, ModelError, ConfigError, RecordError) as exc:
        # Every expected failure mode reaches the user as one line, not a
        # traceback: this is a research tool, not a library.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m probe",
        description="Measure whether a model reproduces a health source's wording.",
    )

    # Carried by every subcommand rather than the top level, so that both
    # "probe --cache-dir X fetch" and the far more natural
    # "probe fetch --cache-dir X" work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    common.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)

    sub = parser.add_subparsers(dest="stage", required=True, parser_class=argparse.ArgumentParser)

    stages = (
        ("fetch", "fetch and extract the source articles", _fetch),
        ("questions", "generate questions from the articles", _questions),
        ("answer", "put each question to the model, no retrieval", _answer),
        ("report", "score the answers, print a table, write JSON", _report),
    )
    for name, help_text, handler in stages:
        stage = sub.add_parser(name, help=help_text, parents=[common])
        stage.set_defaults(handler=handler)

    return parser


def _fetch(args: argparse.Namespace) -> int:
    config = SourceConfig.load(args.config_dir / "source.yaml")
    urls = read_url_list(ROOT / config.url_list)
    cache = ArticleCache(args.cache_dir / "articles")

    articles, failures = fetch_articles(urls, config, cache)

    for failure in failures:
        print(f"warning: {failure}", file=sys.stderr)
    print(f"{len(articles)} article(s) available in {args.cache_dir / 'articles'}")

    if not articles:
        print("error: no articles could be fetched", file=sys.stderr)
        return 1
    return 0


def _questions(args: argparse.Namespace) -> int:
    source = SourceConfig.load(args.config_dir / "source.yaml")
    model = ModelConfig.load(args.config_dir / "model.yaml")
    urls = read_url_list(ROOT / source.url_list)
    articles = ArticleCache(args.cache_dir / "articles").load_all(urls)

    path = args.results_dir / "questions.json"
    records = generate_questions(articles, model, path)
    print(f"{len(records)} question(s) written to {path}")
    return 0


def _answer(args: argparse.Namespace) -> int:
    model = ModelConfig.load(args.config_dir / "model.yaml")
    questions = load_questions(args.results_dir / "questions.json")

    path = args.results_dir / "answers.json"
    records = answer_questions(questions, model, path)
    print(f"{len(records)} answer(s) written to {path}")
    return 0


def _report(args: argparse.Namespace) -> int:
    source = SourceConfig.load(args.config_dir / "source.yaml")
    scoring = ScoringConfig.load(args.config_dir / "scoring.yaml")
    urls = read_url_list(ROOT / source.url_list)
    articles = ArticleCache(args.cache_dir / "articles").load_all(urls)
    answers = load_answers(args.results_dir / "answers.json")

    results = score_answers(answers, articles, scoring)
    path = args.results_dir / "results.json"
    write_results(path, results)

    print(format_summary(results, scoring))
    print(f"\n{len(results)} result(s) written to {path}")
    return 0
