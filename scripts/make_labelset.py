"""Sample 20 scored results across the score range for hand labelling.

Stratified rather than random: a uniform sample of a distribution concentrated
near zero would be twenty near-zero rows and would say nothing about where the
threshold belongs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "question_id",
    "source_url",
    "source_title",
    "question",
    "answer",
    "score",
    "reproduces_source",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "labelled_set.csv")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="fixed so that rerunning selects the same rows as the labels already given",
    )
    args = parser.parse_args(argv)

    try:
        results = json.loads(args.results.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"error: no results at {args.results}; run 'python -m probe report' first", file=sys.stderr)
        return 1

    scorable = [row for row in results if row.get("scorable")]
    if not scorable:
        print("error: no scorable results to sample", file=sys.stderr)
        return 1

    sample = _stratified(scorable, args.rows, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "source_url": row["article_url"],
                    "source_title": row["article_title"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "score": row["score"],
                    # Left empty on purpose. Fill in with yes or no by hand.
                    "reproduces_source": "",
                }
            )

    print(f"{len(sample)} row(s) written to {args.out}")
    print("Fill the reproduces_source column with yes or no, then run scripts/agreement.py")
    return 0


def _stratified(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Spread the sample over equal-width score bands, then fill any shortfall.

    Bands are equal width rather than equal count so the sample follows the
    score range, which is what the threshold question is about.
    """
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: row["score"])
    low, high = ordered[0]["score"], ordered[-1]["score"]

    if high == low:
        return rng.sample(ordered, min(count, len(ordered)))

    bands: list[list[dict]] = [[] for _ in range(count)]
    for row in ordered:
        index = min(int((row["score"] - low) / (high - low) * count), count - 1)
        bands[index].append(row)

    chosen: list[dict] = []
    for band in bands:
        if band:
            chosen.append(rng.choice(band))

    # Equal-width bands are usually sparse at the top, so top the sample back up
    # to `count` from whatever is left, nearest the empty bands first.
    remaining = [row for row in ordered if row not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])

    return sorted(chosen, key=lambda row: row["score"])


if __name__ == "__main__":
    sys.exit(main())
