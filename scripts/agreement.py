"""Report how well the score threshold agrees with hand labels.

Accuracy alone is misleading when one class dominates, which it will if most
answers score near zero, so this also reports the confusion matrix and Cohen's
kappa (agreement above what guessing the base rate would achieve).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe.scoring import ConfigError, ScoringConfig  # noqa: E402

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


@dataclass(frozen=True)
class Agreement:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else float("nan")

    @property
    def recall(self) -> float:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else float("nan")

    @property
    def kappa(self) -> float:
        """Cohen's kappa: 0 is chance agreement, 1 is perfect."""
        total = self.total
        expected = (
            (self.true_positive + self.false_positive) * (self.true_positive + self.false_negative)
            + (self.false_negative + self.true_negative)
            * (self.false_positive + self.true_negative)
        ) / (total * total)
        if expected == 1:
            return float("nan")
        return (self.accuracy - expected) / (1 - expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "results" / "labelled_set.csv")
    parser.add_argument("--scoring", type=Path, default=ROOT / "config" / "scoring.yaml")
    args = parser.parse_args(argv)

    try:
        config = ScoringConfig.load(args.scoring)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    except FileNotFoundError:
        print(f"error: no labelled set at {args.labels}; run scripts/make_labelset.py first", file=sys.stderr)
        return 1

    labelled, skipped = _parse(rows)
    if not labelled:
        print(
            f"error: no labels found in {args.labels}. Fill the reproduces_source "
            "column with yes or no.",
            file=sys.stderr,
        )
        return 1

    agreement = _compare(labelled, config.reuse_threshold)
    print(_format(agreement, config, skipped))
    return 0


def _parse(rows: list[dict[str, str]]) -> tuple[list[tuple[float, bool]], int]:
    """Keep the rows that carry a label; count the rest rather than assuming a value."""
    labelled: list[tuple[float, bool]] = []
    skipped = 0

    for row in rows:
        raw = (row.get("reproduces_source") or "").strip().lower()
        if raw in _YES:
            labelled.append((float(row["score"]), True))
        elif raw in _NO:
            labelled.append((float(row["score"]), False))
        else:
            skipped += 1

    return labelled, skipped


def _compare(labelled: list[tuple[float, bool]], threshold: float) -> Agreement:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for score, label in labelled:
        predicted = score >= threshold
        if predicted and label:
            counts["tp"] += 1
        elif predicted and not label:
            counts["fp"] += 1
        elif not predicted and not label:
            counts["tn"] += 1
        else:
            counts["fn"] += 1

    return Agreement(counts["tp"], counts["fp"], counts["tn"], counts["fn"])


def _format(agreement: Agreement, config: ScoringConfig, skipped: int) -> str:
    lines = [
        f"scoring rule {config.version}, threshold {config.reuse_threshold:g}",
        f"{agreement.total} labelled row(s)" + (f", {skipped} unlabelled skipped" if skipped else ""),
        "",
        f"{'':<22}{'label: yes':>12}{'label: no':>12}",
        f"{'score >= threshold':<22}{agreement.true_positive:>12}{agreement.false_positive:>12}",
        f"{'score <  threshold':<22}{agreement.false_negative:>12}{agreement.true_negative:>12}",
        "",
        f"accuracy  {agreement.accuracy:.3f}",
        f"precision {agreement.precision:.3f}  (of the rows the rule flagged, how many you agreed with)",
        f"recall    {agreement.recall:.3f}  (of the rows you called reuse, how many the rule caught)",
        f"kappa     {agreement.kappa:.3f}  (0 = chance agreement, 1 = perfect)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
