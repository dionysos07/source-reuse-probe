"""Fetch source articles: robots.txt first, rate limited, cached on disk.

The cache is the reason the model stages can be rerun without touching the
source site again, and why the site sees one request per article per project
rather than one per experiment.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml

from probe.extract import ExtractionError, extract_main_text

_REQUIRED_KEYS = (
    "name",
    "url_list",
    "user_agent",
    "request_delay_seconds",
    "timeout_seconds",
    "main_text_selector",
)


class FetchError(Exception):
    """Network, robots or cache failure, reported to the user as a message."""


@dataclass(frozen=True)
class SourceConfig:
    name: str
    url_list: str
    user_agent: str
    request_delay_seconds: float
    timeout_seconds: float
    main_text_selector: str

    @classmethod
    def load(cls, path: Path) -> "SourceConfig":
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise FetchError(f"source config not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise FetchError(f"source config {path} is not valid YAML: {exc}") from exc

        missing = [key for key in _REQUIRED_KEYS if key not in raw]
        if missing:
            raise FetchError(f"source config {path} is missing: {', '.join(missing)}")

        return cls(
            name=str(raw["name"]),
            url_list=str(raw["url_list"]),
            user_agent=str(raw["user_agent"]),
            request_delay_seconds=float(raw["request_delay_seconds"]),
            timeout_seconds=float(raw["timeout_seconds"]),
            main_text_selector=str(raw["main_text_selector"] or ""),
        )


@dataclass(frozen=True)
class Article:
    url: str
    title: str
    text: str
    text_sha256: str
    fetched_at: str


def read_url_list(path: Path) -> list[str]:
    """Read the committed URL list, ignoring blanks and comments."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FetchError(f"URL list not found: {path}") from exc

    urls = [line.strip() for line in lines]
    urls = [url for url in urls if url and not url.startswith("#")]
    if not urls:
        raise FetchError(f"URL list {path} contains no URLs")
    return urls


class ArticleCache:
    """One JSON file per URL, keyed by a hash of the URL.

    Article text lives here and nowhere else in the repo; the directory is
    gitignored so corpus text is never committed.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return self._directory / f"{digest}.json"

    def load(self, url: str) -> Article | None:
        path = self.path_for(url)
        if not path.exists():
            return None
        try:
            return Article(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise FetchError(f"cached article {path} is corrupt; delete it and refetch: {exc}")

    def store(self, article: Article) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(article.url)
        path.write_text(json.dumps(asdict(article), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_all(self, urls: list[str]) -> list[Article]:
        """Every cached article among `urls`, for stages that must not fetch."""
        articles = [self.load(url) for url in urls]
        found = [article for article in articles if article is not None]
        if not found:
            raise FetchError("no articles in the cache; run the fetch stage first")
        return found


class PoliteFetcher:
    """Checks robots.txt once per host and spaces out requests to it."""

    def __init__(self, config: SourceConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers["User-Agent"] = config.user_agent
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._delays: dict[str, float] = {}
        self._last_request: dict[str, float] = {}

    def fetch(self, url: str) -> str:
        """Return the page HTML, or raise FetchError with a usable message."""
        host = urlparse(url).netloc
        if not self._allowed(url, host):
            raise FetchError(f"robots.txt disallows {url} for our user agent")

        self._wait(host)
        try:
            response = self._session.get(url, timeout=self._config.timeout_seconds)
        except requests.Timeout as exc:
            raise FetchError(f"timed out after {self._config.timeout_seconds}s: {url}") from exc
        except requests.RequestException as exc:
            raise FetchError(f"could not fetch {url}: {exc}") from exc
        finally:
            self._last_request[host] = time.monotonic()

        if response.status_code != 200:
            raise FetchError(f"{url} returned HTTP {response.status_code}")

        return response.text

    def _allowed(self, url: str, host: str) -> bool:
        if host not in self._robots:
            self._robots[host] = self._load_robots(url, host)
        return self._robots[host].can_fetch(self._config.user_agent, url)

    def _load_robots(self, url: str, host: str) -> urllib.robotparser.RobotFileParser:
        """Fetch robots.txt through our own session.

        RobotFileParser.read() would open its own connection, bypassing the
        session's user agent and timeout. A site with no robots.txt is treated
        as allowing everything, which is what the standard says.
        """
        parser = urllib.robotparser.RobotFileParser()
        robots_url = urljoin(url, "/robots.txt")
        try:
            response = self._session.get(robots_url, timeout=self._config.timeout_seconds)
        except requests.RequestException as exc:
            raise FetchError(f"could not fetch {robots_url}: {exc}") from exc

        if response.status_code == 200:
            parser.parse(response.text.splitlines())
        else:
            parser.parse([])

        crawl_delay = parser.crawl_delay(self._config.user_agent)
        self._delays[host] = max(
            self._config.request_delay_seconds, float(crawl_delay or 0.0)
        )
        self._last_request[host] = time.monotonic()
        return parser

    def _wait(self, host: str) -> None:
        delay = self._delays.get(host, self._config.request_delay_seconds)
        last = self._last_request.get(host)
        if last is None:
            return
        remaining = delay - (time.monotonic() - last)
        if remaining > 0:
            time.sleep(remaining)


def fetch_articles(
    urls: list[str], config: SourceConfig, cache: ArticleCache
) -> tuple[list[Article], list[str]]:
    """Fetch and extract every URL, skipping ones already cached.

    Returns the articles and a list of human-readable failures: one bad URL
    should not abandon the other eleven.
    """
    fetcher: PoliteFetcher | None = None
    articles: list[Article] = []
    failures: list[str] = []

    for url in urls:
        cached = cache.load(url)
        if cached is not None:
            articles.append(cached)
            continue

        if fetcher is None:
            fetcher = PoliteFetcher(config)

        try:
            html = fetcher.fetch(url)
            extracted = extract_main_text(html, config.main_text_selector or None)
        except (FetchError, ExtractionError) as exc:
            failures.append(f"{url}: {exc}")
            continue

        article = Article(
            url=url,
            title=extracted.title,
            text=extracted.text,
            text_sha256=hashlib.sha256(extracted.text.encode("utf-8")).hexdigest(),
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        cache.store(article)
        articles.append(article)

    return articles, failures
