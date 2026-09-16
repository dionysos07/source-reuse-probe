import pytest

from probe.extract import ExtractionError, extract_main_text, normalise_whitespace


def test_semantic_article_keeps_prose_and_drops_chrome(fixture_html):
    article = extract_main_text(fixture_html("article_semantic.html"))

    assert article.title == "Gout"
    assert "common form of arthritis" in article.text
    assert "Intense joint pain" in article.text

    # Navigation, scripts, the aside and the footer must not reach the scorer:
    # text shared by every page would inflate every score equally.
    assert "Health Topics" not in article.text
    assert "tracking" not in article.text
    assert "Related: Arthritis" not in article.text
    assert "All rights reserved" not in article.text
    assert "Page last updated" not in article.text


def test_nested_block_is_not_counted_twice(fixture_html):
    article = extract_main_text(fixture_html("article_semantic.html"))
    assert article.text.count("Redness and tenderness") == 1


def test_nbsp_becomes_a_plain_space(fixture_html):
    article = extract_main_text(fixture_html("article_semantic.html"))
    assert "uric acid" in article.text
    assert "\xa0" not in article.text


def test_div_soup_falls_back_to_the_densest_block(fixture_html):
    article = extract_main_text(fixture_html("article_divsoup.html"))

    assert article.title == "Shingles"
    assert "varicella zoster virus" in article.text
    # The sidebar loses to the content div on paragraph text.
    assert "Browse topics" not in article.text


def test_selector_wins_over_the_fallback_chain(fixture_html):
    article = extract_main_text(fixture_html("article_divsoup.html"), selector="#sidebar")
    assert "Browse topics" in article.text
    assert "varicella" not in article.text


def test_unknown_selector_falls_back_rather_than_failing(fixture_html):
    article = extract_main_text(fixture_html("article_divsoup.html"), selector=".does-not-exist")
    assert "varicella zoster virus" in article.text


def test_umlauts_survive_extraction(fixture_html):
    article = extract_main_text(fixture_html("article_unicode.html"))

    assert article.title == "Gicht und Ernährung"
    assert "Gemüse gehören" in article.text
    assert "Straße" in article.text


def test_decomposed_umlaut_is_normalised_to_composed_form(fixture_html):
    article = extract_main_text(fixture_html("article_unicode.html"))
    # The page writes "Gefa" + combining diaeresis; a model answer would write
    # the precomposed character. They must compare equal.
    assert "Gefäße" in article.text
    assert "ä" not in article.text


def test_page_without_prose_raises_a_named_error(fixture_html):
    with pytest.raises(ExtractionError):
        extract_main_text(fixture_html("article_empty.html"))


def test_empty_input_raises_rather_than_returning_empty():
    with pytest.raises(ExtractionError):
        extract_main_text("")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  a   b  ", "a b"),
        ("a\n\n  b  ", "a\n\nb"),
        ("a\xa0b", "a b"),
        ("", ""),
        ("Ä̈", "Ä̈"),
    ],
)
def test_normalise_whitespace_cases(raw, expected):
    assert normalise_whitespace(raw) == expected
