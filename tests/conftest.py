from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_html():
    """Read a fixture by name. No network in any test in this suite."""

    def _read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def fixture_text():
    def _read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def scoring_config():
    """The real config/scoring.yaml, so the tests fail if the shipped rule breaks."""
    from probe.scoring import ScoringConfig

    return ScoringConfig.load(Path(__file__).parents[1] / "config" / "scoring.yaml")
