# Evaluation Report

## Accuracy on labeled sample set (final strategy)

- Matched rows: 5
- Missing predictions: 15

| Field | Metric | Score | n |
|---|---|---|---|
| evidence_standard_met | exact_match_accuracy | 0.8 | 5 |
| claim_status | exact_match_accuracy | 0.6 | 5 |
| issue_type | exact_match_accuracy | 0.4 | 5 |
| object_part | exact_match_accuracy | 1.0 | 5 |
| severity | exact_match_accuracy | 0.4 | 5 |
| valid_image | exact_match_accuracy | 1.0 | 5 |
| risk_flags | avg_jaccard_overlap | 0.667 | 5 |
| supporting_image_ids | avg_jaccard_overlap | 0.9 | 5 |

## Strategy comparison

| Field | gemini-2.5-flash | gemini-2.5-flash-lite v1 | gemini-2.5-flash-lite v2 (severity-calibrated) |
|---|---|---|---|
| evidence_standard_met | 0.8 | 0.8 | 0.8 |
| claim_status | 0.6 | 0.6 | 0.6 |
| issue_type | 0.4 | 0.4 | 0.4 |
| object_part | 1.0 | 1.0 | 1.0 |
| severity | 0.2 | 0.2 | 0.4 |
| valid_image | 1.0 | 1.0 | 1.0 |
| risk_flags | 0.517 | 0.667 | 0.667 |
| supporting_image_ids | 0.6 | 0.9 | 0.9 |

**Operational comparison**

| Metric | gemini-2.5-flash | gemini-2.5-flash-lite v1 | gemini-2.5-flash-lite v2 (severity-calibrated) |
|---|---|---|---|
| call_count | 5 | 5 | 5 |
| failed_calls | 0 | 0 | 0 |
| input_tokens | 12029 | 12029 | 13284 |
| output_tokens | 901 | 951 | 904 |
| elapsed_seconds | 53.56 | 24.88 | 23.0 |


## Operational analysis

### Sample set processing
- Model calls: 5
- Failed calls: 0
- Retried calls: 0
- Images processed: 8
- Input tokens: 13284
- Output tokens: 904
- Runtime: 23.0s
- Estimated cost: $0.0

### Test set processing (projected from sample, scaled x8.80 for 44 rows)
- Projected model calls: 44
- Projected images processed: 70
- Projected input tokens: 116899
- Projected output tokens: 7955
- Projected runtime: ~202s (serialized; lower with batching/parallelism)
- Projected cost: $0.0

### Rate limits, batching, and reliability strategy
- One Gemini call per claim row (1 claim = 1 call, all of that claim's images attached together) — this keeps reasoning grounded across all images for a claim at once, instead of fragmenting context per image.
- Calls are paced with a fixed minimum interval (5.0s) between requests to stay under free-tier RPM limits.
- Transient/rate-limit failures are retried with exponential backoff, up to 4 attempts, before falling back to a safe default (not_enough_information + manual_review_required) rather than crashing the run.
- Claims with no loadable images skip the API call entirely (no point spending a call on unusable input) and are marked not_enough_information directly.
- No response caching is implemented since claims are processed once each; if reruns become common, caching by (user_claim, image_paths) hash would avoid redundant calls.

## Known limitations and next steps

This evaluation was run on a 5-claim subset of the labeled sample set due to
free-tier daily quota constraints (20 requests/day on this account, shared
project-wide rather than per-model). Two real prompt issues were identified
and fixed through this iteration:

1. **Severity calibration** — the initial prompt left "low/medium/high"
   undefined, causing the model to over-rate severity (e.g. marking a single
   dent as "high"). Adding explicit calibration criteria (what counts as
   low/medium/high per type of damage) improved severity accuracy from
   1/5 to 2/5 on this batch.

2. **Cross-image consistency** — an early version of the prompt didn't
   instruct the model to verify all images in a multi-image claim show the
   same physical object, causing false "contradicted" results when one
   photo simply didn't capture the damage from its angle. This was fixed
   with explicit cross-image consistency guidance.

**Remaining known weakness:** vehicle/object identity verification across
multiple images is still unreliable in edge cases (e.g. one claim where two
images may show different vehicles was scored as "supported" instead of
"not_enough_information"). The model correctly follows the evidence rules
it's given, but fine-grained visual identity-matching across photos is a
genuine capability limit of fast-tier vision models, not a prompting gap.
A stronger model, or a dedicated image-similarity pre-check before the main
reasoning call, would likely close this gap — left as a future improvement
given time and quota constraints for this submission.

Given the small sample size (5 of 19 labeled examples), these accuracy
numbers should be read as directional rather than statistically robust;
the full sample set evaluation (when quota allows) would give a more
reliable picture.