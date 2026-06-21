# Evaluation Report

## Accuracy on labeled sample set (final strategy)

- Matched rows: 44
- Missing predictions: 0

| Field | Metric | Score | n |
|---|---|---|---|
| evidence_standard_met | exact_match_accuracy | 0.0 | 44 |
| claim_status | exact_match_accuracy | 0.0 | 44 |
| issue_type | exact_match_accuracy | 0.0 | 44 |
| object_part | exact_match_accuracy | 0.0 | 44 |
| severity | exact_match_accuracy | 0.0 | 44 |
| valid_image | exact_match_accuracy | 0.0 | 44 |
| risk_flags | avg_jaccard_overlap | 0.0 | 44 |
| supporting_image_ids | avg_jaccard_overlap | 0.0 | 44 |

## Operational analysis

### Sample set processing
(no run_stats file provided)

### Test set processing
(no stats available — run run_pipeline.py against the test set to populate this)

### Rate limits, batching, and reliability strategy
- One Gemini call per claim row (1 claim = 1 call, all of that claim's images attached together) — this keeps reasoning grounded across all images for a claim at once, instead of fragmenting context per image.
- Calls are paced with a fixed minimum interval (5.0s) between requests to stay under free-tier RPM limits.
- Transient/rate-limit failures are retried with exponential backoff, up to 4 attempts, before falling back to a safe default (not_enough_information + manual_review_required) rather than crashing the run.
- Claims with no loadable images skip the API call entirely (no point spending a call on unusable input) and are marked not_enough_information directly.
- No response caching is implemented since claims are processed once each; if reruns become common, caching by (user_claim, image_paths) hash would avoid redundant calls.
