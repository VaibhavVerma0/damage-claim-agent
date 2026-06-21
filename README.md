# Multi-Modal Evidence Review Agent

Automated damage-claim reviewer. Given a user's claim conversation, submitted
photos, claim history, and minimum evidence rules, it decides whether the
photo evidence supports, contradicts, or fails to address the claim.

## How it works

1. For each claim row, load all referenced images.
2. Build a prompt containing: the claim conversation, the relevant evidence
   requirements for that object type, and the user's claim history.
3. Send the images + prompt to Gemini 2.0 Flash with forced JSON output
   matching a strict schema.
4. Validate every field against the allowed-value lists from the problem
   statement, clamping or falling back to a safe default
   (`unknown` / `not_enough_information`) if the model returns something
   off-spec.
5. Write one row to `output.csv` per claim, in the required column order.

Why this design:
- **One call per claim, all images attached together** — lets the model
  reason across all of a claim's photos at once instead of losing context
  per-image.
- **Forced JSON schema** — the output columns have strict allowed values;
  free-text generation would need brittle parsing.
- **Validation layer after the model call** — even with JSON mode, models
  occasionally drift outside the allowed lists; this guarantees the final
  CSV is always schema-valid.
- **No-image claims skip the API call** — saves cost on input that can't be
  evaluated anyway.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real key
export GEMINI_API_KEY=your-key-here   # or `source .env` depending on your shell
```

Get a free key at https://aistudio.google.com/apikey — no credit card
required for the free tier.

## Step 1 — sanity check on the labeled sample set

```bash
python run_pipeline.py \
  --claims dataset/sample_claims.csv \
  --user-history dataset/user_history.csv \
  --evidence dataset/evidence_requirements.csv \
  --images-root dataset \
  --output sample_output.csv \
  --stats-out evaluation/run_stats_sample.json
```

Then check accuracy against the known answers and generate the operational
report:

```bash
python evaluation/evaluate.py \
  --gold dataset/sample_claims.csv \
  --predicted sample_output.csv \
  --sample-stats evaluation/run_stats_sample.json \
  --test-row-count <number of rows in dataset/claims.csv> \
  --cost-per-1k-input-tokens <check current Gemini pricing> \
  --cost-per-1k-output-tokens <check current Gemini pricing> \
  --out evaluation/evaluation_report.md
```

`--gold` is assumed to use the same column names as `output.csv` for the
expected answers. If your real `sample_claims.csv` names those columns
differently, set `GOLD_COLUMN_MAP` near the top of `evaluation/evaluate.py`.

## Step 2 — run on the real test set

```bash
python run_pipeline.py \
  --claims dataset/claims.csv \
  --user-history dataset/user_history.csv \
  --evidence dataset/evidence_requirements.csv \
  --images-root dataset \
  --output output.csv \
  --stats-out evaluation/run_stats_test.json
```

Then re-run `evaluation/evaluate.py` with `--test-stats
evaluation/run_stats_test.json` so the report uses real numbers instead of
projections from the sample set.

## Tuning

- `config.MIN_SECONDS_BETWEEN_CALLS` — pacing between API calls, to stay
  under your free-tier rate limit. Lower it if your quota allows; raise it
  if you hit 429 errors.
- `config.MAX_RETRIES` / `RETRY_BACKOFF_SECONDS` — retry behavior on
  transient failures.
- `--limit N` on `run_pipeline.py` — process only the first N rows, handy
  for a quick smoke test before running the full set.

## Files

```
config.py             Allowed-value lists, model name, retry/pacing settings
io_utils.py            CSV and image loading helpers
gemini_client.py        Gemini API wrapper: retries, pacing, usage tracking
claim_agent.py          Prompt construction + response validation
run_pipeline.py          Entry point: run the agent over a claims CSV
evaluation/evaluate.py   Accuracy scoring + operational report generator
```
