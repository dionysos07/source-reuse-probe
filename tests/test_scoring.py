from dataclasses import replace
from pathlib import Path

import pytest

from probe.scoring import ConfigError, ScoringConfig, normalise, score_overlap


def test_config_loads_the_shipped_rule(scoring_config):
    assert scoring_config.version == "v1"
    assert scoring_config.ngram_size == 4
    assert "the" in scoring_config.stopwords
    assert "gout" not in scoring_config.stopwords


def test_missing_config_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ScoringConfig.load(tmp_path / "absent.yaml")


def test_incomplete_config_names_the_missing_keys(tmp_path):
    path = tmp_path / "scoring.yaml"
    path.write_text("version: v9\nngram_size: 4\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        ScoringConfig.load(path)


def test_normalise_drops_stopwords_and_punctuation(scoring_config):
    assert normalise("The pain, and the swelling!", scoring_config) == ["pain", "swelling"]


def test_normalise_keeps_contractions_whole(scoring_config):
    # "don't" is in the stopword list; splitting it would leave "don" behind.
    assert normalise("Don’t ignore the pain", scoring_config) == ["ignore", "pain"]


def test_normalise_casefolds_sharp_s(scoring_config):
    assert normalise("Straße", scoring_config) == normalise("STRASSE", scoring_config)


def test_answer_identical_to_source_scores_one(scoring_config, fixture_text):
    source = fixture_text("source_gout.txt")
    result = score_overlap(source, source, scoring_config)

    assert result.score == 1.0
    assert result.matched == result.total > 0
    assert result.config_version == "v1"


def test_answer_with_no_overlap_scores_zero(scoring_config, fixture_text):
    result = score_overlap(
        "Volcanoes erupt when magma rises through fractures in the crust.",
        fixture_text("source_gout.txt"),
        scoring_config,
    )
    assert result.score == 0.0
    assert result.matched == 0
    assert result.total > 0
    assert result.scorable


def test_empty_answer_is_marked_unscorable_rather_than_zero(scoring_config, fixture_text):
    result = score_overlap("", fixture_text("source_gout.txt"), scoring_config)

    assert result.total == 0
    assert result.score == 0.0
    assert not result.scorable


def test_answer_too_short_for_one_ngram_is_unscorable(scoring_config, fixture_text):
    # Three content words cannot form a 4-gram.
    result = score_overlap("Gout causes pain.", fixture_text("source_gout.txt"), scoring_config)
    assert not result.scorable


def test_verbatim_sentence_from_the_source_scores_one(scoring_config, fixture_text):
    result = score_overlap(
        "It happens when uric acid builds up in the blood and forms crystals in a joint.",
        fixture_text("source_gout.txt"),
        scoring_config,
    )
    assert result.score == 1.0


def test_partial_reuse_scores_between_zero_and_one(scoring_config, fixture_text):
    answer = (
        "Gout is a common form of arthritis that causes sudden pain and swelling. "
        "Bananas are harvested year round in tropical regions far from any clinic."
    )
    result = score_overlap(answer, fixture_text("source_gout.txt"), scoring_config)
    assert 0.0 < result.score < 1.0


def test_repetition_adds_n_grams_and_a_boundary_penalty(scoring_config, fixture_text):
    """Pins two consequences of the v1 rule that are easy to mistake for bugs.

    N-grams are counted with multiplicity, so repeating a matching sentence adds
    matches rather than collapsing them. But v1 knows nothing about sentence
    boundaries, so the join between the two copies produces n-grams that exist in
    neither the source nor any sentence of the answer, and the score falls below
    the single-sentence case. Both are documented in the README.
    """
    sentence = "Attacks often begin at night and most often affect the big toe."
    source = fixture_text("source_gout.txt")

    once = score_overlap(sentence, source, scoring_config)
    twice = score_overlap(sentence + " " + sentence, source, scoring_config)

    assert once.score == 1.0
    assert twice.matched > once.matched
    assert twice.total == 2 * once.total + scoring_config.ngram_size - 1
    assert twice.score < once.score


def test_umlauts_match_when_spelled_the_same(scoring_config, fixture_text):
    result = score_overlap(
        "Die Ernährung bei Gicht sollte wenig Fleisch enthalten.",
        fixture_text("source_umlaut.txt"),
        scoring_config,
    )
    assert result.score == 1.0


def test_umlauts_are_not_folded_away_by_default(scoring_config, fixture_text):
    """Baren must not match Bären: v1 keeps diacritics precisely to avoid this."""
    stripped = replace(scoring_config, strip_diacritics=True, ngram_size=1)
    kept = replace(scoring_config, ngram_size=1)
    source = fixture_text("source_umlaut.txt")

    assert score_overlap("Baren", source, kept).score == 1.0
    assert normalise("Bären", kept) != normalise("Baren", kept)
    assert normalise("Bären", stripped) == normalise("Baren", stripped)


def test_decomposed_and_composed_umlauts_compare_equal(scoring_config):
    composed = normalise("Gefäße", scoring_config)
    decomposed = normalise("Gefäße", scoring_config)
    assert composed == decomposed


def test_source_with_no_content_words_yields_zero(scoring_config):
    result = score_overlap("Gout causes sudden joint pain", "the and of to", scoring_config)
    assert result.score == 0.0
    assert result.scorable
