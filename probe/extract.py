"""Turn a fetched HTML page into the article prose we score against.

The generic path here exists because the build environment for this repo has no
route to the source site, so a hand-written site-specific CSS selector could not
be verified against a live page. Set `main_text_selector` in config/source.yaml
once you can check one; it takes priority over everything below.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Tag

# Chrome, navigation and interactive furniture. Present on every page of a site
# and therefore pure noise for a reuse measurement: if every article "shares"
# the site's footer with every answer, the score stops meaning anything.
_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "iframe",
    "svg",
    "figure",
)

# Elements whose text is a paragraph-like unit. Kept as separate lines so that
# list items do not run into each other and produce n-grams that span a boundary
# no human would read as continuous prose.
_BLOCK_TAGS = ("p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "dt", "dd", "td")

_WHITESPACE = re.compile(r"[^\S\n]+")

# How much of the best paragraph-text score a deeper container must retain to be
# preferred over its parent. See _densest_block.
RETAIN_FRACTION = 0.8


class ExtractionError(Exception):
    """Raised when a page yields no usable prose, so the CLI can say which URL."""


@dataclass(frozen=True)
class ExtractedArticle:
    title: str
    text: str


def extract_main_text(html: str, selector: str | None = None) -> ExtractedArticle:
    """Extract the article title and body prose from a full HTML page.

    `selector` is a CSS selector for the article body. When it is empty or
    matches nothing we fall back to <article>, then <main>, then the container
    holding the most paragraph text, in that order.

    Raises ExtractionError if nothing readable is left.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    container = _select_container(soup, selector)
    if container is None:
        raise ExtractionError("no document body in page")

    text = _block_text(container)
    if not text:
        raise ExtractionError("page has no extractable prose")

    return ExtractedArticle(title=_title(soup, container), text=text)


def _select_container(soup: BeautifulSoup, selector: str | None) -> Tag | None:
    """Pick the element holding the article body, cheapest reliable signal first."""
    if selector:
        chosen = soup.select_one(selector)
        if chosen is not None:
            return chosen

    for tag_name in ("article", "main"):
        chosen = soup.find(tag_name)
        if isinstance(chosen, Tag):
            return chosen

    return _densest_block(soup) or soup.body or soup


def _densest_block(soup: BeautifulSoup) -> Tag | None:
    """The tightest element that still holds nearly all of the paragraph text.

    Plain "most text wins" can only ever pick the outermost wrapper, since every
    ancestor contains at least as much text as its child. So we take the deepest
    element that retains RETAIN_FRACTION of the best score: descending past the
    article body costs real prose, while descending past a wrapper that merely
    adds a sidebar costs almost none. A page whose sidebar rivals the article in
    length defeats this and keeps both; that is a known weakness, documented in
    the README.
    """
    scored: list[tuple[int, int, Tag]] = []

    for candidate in soup.find_all(("div", "section", "body")):
        length = sum(len(p.get_text(" ", strip=True)) for p in candidate.find_all("p"))
        if length > 0:
            scored.append((length, len(list(candidate.parents)), candidate))

    if not scored:
        return None

    cutoff = max(length for length, _, _ in scored) * RETAIN_FRACTION
    eligible = [item for item in scored if item[0] >= cutoff]
    _, _, best = max(eligible, key=lambda item: (item[1], item[0]))
    return best


def _block_text(container: Tag) -> str:
    """Join block elements with newlines, in document order, de-duplicated.

    Nested blocks (a <p> inside an <li>) would otherwise contribute their text
    twice and double-count against the n-gram overlap.
    """
    lines: list[str] = []
    seen: set[int] = set()

    for block in container.find_all(_BLOCK_TAGS):
        if any(id(parent) in seen for parent in block.parents):
            continue
        line = normalise_whitespace(block.get_text(" ", strip=True))
        if line:
            seen.add(id(block))
            lines.append(line)

    if not lines:
        # A page whose prose sits in bare divs with no block markup at all.
        return normalise_whitespace(container.get_text(" ", strip=True))

    return "\n".join(lines)


def _title(soup: BeautifulSoup, container: Tag) -> str:
    heading = container.find("h1") or soup.find("h1") or soup.find("title")
    if heading is None:
        return ""
    return normalise_whitespace(heading.get_text(" ", strip=True))


def normalise_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace, keeping newlines and umlauts.

    NFC first so that a decomposed "a + combining diaeresis" from the page and a
    precomposed "ä" from a model answer compare equal downstream.
    """
    text = unicodedata.normalize("NFC", text.replace("\xa0", " ").replace("​", ""))
    text = _WHITESPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()
