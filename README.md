# source-reuse-probe

A research prototype that measures how much of a trusted health source's wording
turns up in a language model's answers to questions that source could answer.

## What this is

The pipeline has four stages, each a separate command so that re-scoring never
costs another model call:

1. **fetch** — read a committed list of article URLs from one source site, check
   `robots.txt`, fetch at a rate limit, extract the article prose, cache it on
   disk.
2. **questions** — ask the model, given an article, for questions that article
   answers.
3. **answer** — put each question back to the model on its own: no retrieval, no
   mention of the source, no hint that a specific article exists.
4. **report** — score every answer against its source article and print a table.

The source is [MedlinePlus](https://medlineplus.gov/), published by the U.S.
National Library of Medicine. It was chosen over other trusted health sites
because its content is in the public domain, so caching article text locally and
publishing derived numbers raises no licensing question.

Scoring rule v1, which lives in `config/scoring.yaml` and not in Python:
normalise case and whitespace, strip stopwords, take content-word n-grams of
length 4, and score the fraction of the answer's n-grams that appear in the
source article. Changing the rule is a diff in that file.

Every stored result records the requested model id, the model id the API
actually served, the exact system and user prompts, the scoring config version,
and UTC timestamps for the answer and the scoring.

## What it does not do

- It does not query AI Overviews, Google, or any search engine. See the next
  section.
- It does not prove that an answer was derived from the source. It measures
  string overlap, which is evidence, not attribution.
- It does not measure paraphrase. An answer that reproduces the source's
  substance in different words scores near zero.
- There is no API, dashboard, database, container, or scheduled job. Results are
  files on disk.
- It supports one scoring rule and one model provider. Swapping either is an
  edit to the code, deliberately.
- Article text is never committed. `cache/` is gitignored; only the URL list and
  derived numbers are in the repository.

## How to run it

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # never read from a file in this repo

python -m probe fetch                 # network; writes cache/articles/
python -m probe questions             # model calls; writes results/questions.json
python -m probe answer                # model calls; writes results/answers.json
python -m probe report                # no network; prints a table, writes results/results.json
```

Then the labelled set:

```sh
python scripts/make_labelset.py       # writes results/labelled_set.csv, 20 rows
# fill in the reproduces_source column by hand with yes or no
python scripts/agreement.py           # compares those labels against the threshold
```

Tests, which touch neither the network nor the model:

```sh
pytest
```

`results/` is empty until you run the pipeline; no run output is committed.

Configuration lives in `config/`: `source.yaml` (site, user agent, rate limit,
article selector), `model.yaml` (model id, token cap, questions per article),
`scoring.yaml` (the rule and the reuse threshold). Paths are resolved relative to
the repository; every stage accepts `--config-dir`, `--cache-dir` and
`--results-dir` to override them.

## The substitution of a directly-called model for a search engine

The question this prototype is built for is whether trusted health content is
reproduced in AI-generated search answers. It does not measure that. It measures
whether a directly-called model reproduces that content.

AI Overviews has no API, and scraping it is both unreliable and against Google's
terms, so the search engine is replaced by a model called straight through the
Messages API. This is a substitution, not an approximation, and it differs from
the real system in at least these ways:

- **No retrieval.** AI Overviews generates its answer with search results in
  context and frequently quotes them. This prototype gives the model nothing but
  the question, so it can only reproduce what is in its weights. Overlap
  measured here is a *floor*, and a floor of a different mechanism.
- **Different model.** The model behind AI Overviews is not the model called
  here, and it is prompted, tuned and post-processed differently.
- **No ranking.** A real answer depends on which pages ranked for that query on
  that day. Here the source article is fixed in advance.
- **Different answer shape.** Search answers are short, formatted for a results
  page, and carry link cards. Nothing here reproduces that.

What the prototype does do is exercise the measurement end of the problem — URL
list to fetch to question to answer to score to hand-labelled agreement — on a
substrate that can actually be run. The scoring, extraction and labelling code
would not change if the answer source did.

## Known weaknesses of the scoring rule

- **It measures density, not amount.** The score is the fraction of the
  *answer's* n-grams found in the source. A one-sentence answer lifted verbatim
  scores 1.0; a long answer containing that same lifted sentence among ten
  paragraphs of its own prose scores low. Two answers reproducing exactly as
  much source text can score an order of magnitude apart.
- **No control corpus.** Health writing is formulaic. "Talk to your doctor if
  symptoms persist" appears on every health site there is, so overlap with this
  source is not evidence of reuse *of this source*. Without scoring the same
  answers against unrelated articles there is no baseline to subtract.
- **Paraphrase is invisible.** Reproducing a source's substance in new words
  scores zero. If a model has learned to paraphrase rather than quote, this rule
  reports no reuse where a reader would see plenty.
- **Stopword stripping joins text that was never adjacent.** After stopwords are
  removed, "pain and swelling in the joints" becomes `pain swelling joints`, so
  an answer that names those three things in an unrelated construction can match
  a 4-gram that does not exist as contiguous prose in either text.
- **N-grams span sentence and paragraph boundaries.** The rule has no notion of a
  sentence, so a 4-gram can straddle the join between two sentences. This creates
  matches no reader would call quotation, and also *penalises* answers that
  repeat themselves, because the join produces n-grams found nowhere.
  `tests/test_scoring.py` pins both effects.
- **Short answers are unscorable, not zero.** An answer with fewer than four
  content words forms no n-gram at all. Those are reported separately rather
  than averaged in as zeroes, which would understate the mean.
- **Extraction noise inflates scores.** Boilerplate shared by every page of the
  site would overlap with every answer equally. Navigation, headers, footers and
  asides are stripped, but the fallback that picks the article container by
  paragraph density can keep a sidebar when the sidebar is nearly as long as the
  article. `config/source.yaml` takes a site-specific CSS selector to avoid the
  heuristic entirely; it ships empty because it could not be verified against a
  live page from the environment this code was written in.
- **The threshold is reasoned, not observed.** `reuse_threshold` in
  `config/scoring.yaml` decides what counts as reuse. Its value is derived from
  the rule's own arithmetic — the coincidence floor is effectively zero, and one
  verbatim sentence scores three times higher in a short answer than in a long
  one — not from a measured score distribution. `scripts/agreement.py` exists to
  check it against hand labels, and the config records what would justify moving
  it.
- **One rule, one length.** n = 4 is a judgement call. Shorter n matches common
  phrasing and inflates scores; longer n catches only long verbatim runs.
