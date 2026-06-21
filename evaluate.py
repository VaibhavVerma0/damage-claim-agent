"""
Run this AFTER you've used run_pipeline.py to produce predictions for
dataset/sample_claims.csv (the labeled set).

Usage:
    python evaluation/evaluate.py \
        --gold dataset/sample_claims.csv \
        --predicted sample_output.csv \
        --sample-stats evaluation/run_stats_sample.json \
        --test-stats evaluation/run_stats_test.json \
        --test-row-count 500 \
        --cost-per-1k-input-tokens 0.0 \
        --cost-per-1k-output-tokens 0.0 \
        --out evaluation/evaluation_report.md

NOTE: --gold is assumed to have the same column names as output.csv
(user_id, image_paths, user_claim, claim_object, plus all the answer
columns). If your real sample_claims.csv uses different column names for
the expected answers, adjust GOLD_COLUMN_MAP below.
"""

import argparse
import csv
import json
import sys

sys.path.insert(0, ".")
import config  # noqa: E402

EXACT_MATCH_FIELDS = [
    "evidence_standard_met", "claim_status", "issue_type",
    "object_part", "severity", "valid_image",
]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]

# If your gold CSV names the answer columns differently, map them here:
# {gold_column_name: output_column_name}
GOLD_COLUMN_MAP = {}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_str(v):
    return str(v).strip().lower()


def set_from_field(v):
    if not v:
        return set()
    return {p.strip() for p in str(v).split(";") if p.strip()}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def evaluate(gold_rows, pred_rows):
    pred_by_key = {}
    for r in pred_rows:
        key = (r.get("user_id"), r.get("image_paths"))
        pred_by_key[key] = r

    exact_scores = {f: [] for f in EXACT_MATCH_FIELDS}
    set_scores = {f: [] for f in SET_FIELDS}
    matched, missing = 0, 0

    for gold in gold_rows:
        for gold_col, out_col in GOLD_COLUMN_MAP.items():
            if gold_col in gold:
                gold[out_col] = gold[gold_col]

        key = (gold.get("user_id"), gold.get("image_paths"))
        pred = pred_by_key.get(key)
        if pred is None:
            missing += 1
            continue
        matched += 1

        for field in EXACT_MATCH_FIELDS:
            g, p = normalize_str(gold.get(field, "")), normalize_str(pred.get(field, ""))
            exact_scores[field].append(1 if g == p else 0)

        for field in SET_FIELDS:
            g_set = set_from_field(gold.get(field, ""))
            p_set = set_from_field(pred.get(field, ""))
            set_scores[field].append(jaccard(g_set, p_set))

    report = {"matched_rows": matched, "missing_predictions": missing, "fields": {}}
    for field, scores in exact_scores.items():
        report["fields"][field] = {
            "type": "exact_match_accuracy",
            "score": round(sum(scores) / len(scores), 3) if scores else None,
            "n": len(scores),
        }
    for field, scores in set_scores.items():
        report["fields"][field] = {
            "type": "avg_jaccard_overlap",
            "score": round(sum(scores) / len(scores), 3) if scores else None,
            "n": len(scores),
        }
    return report


def compare_strategies(gold_rows, strategy_predictions):
    """
    strategy_predictions: list of (label, pred_rows, stats_dict_or_None) tuples.
    Returns a list of (label, accuracy_report, stats) for a side-by-side table.
    """
    results = []
    for label, pred_rows, stats in strategy_predictions:
        report = evaluate(gold_rows, pred_rows)
        results.append((label, report, stats))
    return results


def write_comparison_section(lines, comparison_results):
    lines.append("\n## Strategy comparison\n")
    if not comparison_results:
        return
    field_names = list(comparison_results[0][1]["fields"].keys())
    header = "| Field | " + " | ".join(label for label, _, _ in comparison_results) + " |"
    divider = "|---|" + "---|" * len(comparison_results)
    lines.append(header)
    lines.append(divider)
    for field in field_names:
        row = [field]
        for _, report, _ in comparison_results:
            row.append(str(report["fields"][field]["score"]))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n**Operational comparison**\n")
    lines.append("| Metric | " + " | ".join(label for label, _, _ in comparison_results) + " |")
    lines.append("|---|" + "---|" * len(comparison_results))
    metrics = ["call_count", "failed_calls", "input_tokens", "output_tokens", "elapsed_seconds"]
    for metric in metrics:
        row = [metric]
        for _, _, stats in comparison_results:
            row.append(str(stats.get(metric, "n/a")) if stats else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def load_stats(path):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def estimate_cost(stats, cost_in_per_1k, cost_out_per_1k):
    if not stats:
        return None
    cost = (stats["input_tokens"] / 1000) * cost_in_per_1k + \
           (stats["output_tokens"] / 1000) * cost_out_per_1k
    return round(cost, 4)


def write_report(path, accuracy_report, sample_stats, test_stats, test_row_count,
                  cost_in_per_1k, cost_out_per_1k, comparison_results=None):
    lines = []
    lines.append("# Evaluation Report\n")

    lines.append("## Accuracy on labeled sample set (final strategy)\n")
    lines.append(f"- Matched rows: {accuracy_report['matched_rows']}")
    lines.append(f"- Missing predictions: {accuracy_report['missing_predictions']}\n")
    lines.append("| Field | Metric | Score | n |")
    lines.append("|---|---|---|---|")
    for field, info in accuracy_report["fields"].items():
        lines.append(f"| {field} | {info['type']} | {info['score']} | {info['n']} |")

    if comparison_results:
        write_comparison_section(lines, comparison_results)

    lines.append("\n## Operational analysis\n")

    if sample_stats:
        lines.append("### Sample set processing")
        lines.append(f"- Model calls: {sample_stats['call_count']}")
        lines.append(f"- Failed calls: {sample_stats['failed_calls']}")
        lines.append(f"- Retried calls: {sample_stats['retried_calls']}")
        lines.append(f"- Images processed: {sample_stats['images_processed']}")
        lines.append(f"- Input tokens: {sample_stats['input_tokens']}")
        lines.append(f"- Output tokens: {sample_stats['output_tokens']}")
        lines.append(f"- Runtime: {sample_stats['elapsed_seconds']}s")
        sample_cost = estimate_cost(sample_stats, cost_in_per_1k, cost_out_per_1k)
        lines.append(f"- Estimated cost: ${sample_cost}\n")
    else:
        lines.append("### Sample set processing\n(no run_stats file provided)\n")

    if test_stats:
        lines.append("### Test set processing (actual)")
        lines.append(f"- Model calls: {test_stats['call_count']}")
        lines.append(f"- Failed calls: {test_stats['failed_calls']}")
        lines.append(f"- Retried calls: {test_stats['retried_calls']}")
        lines.append(f"- Images processed: {test_stats['images_processed']}")
        lines.append(f"- Input tokens: {test_stats['input_tokens']}")
        lines.append(f"- Output tokens: {test_stats['output_tokens']}")
        lines.append(f"- Runtime: {test_stats['elapsed_seconds']}s")
        test_cost = estimate_cost(test_stats, cost_in_per_1k, cost_out_per_1k)
        lines.append(f"- Estimated cost: ${test_cost}\n")
    elif sample_stats and test_row_count:
        # Project from sample stats if we don't have real test stats yet
        n_sample = max(accuracy_report["matched_rows"], 1)
        scale = test_row_count / n_sample
        projected_calls = round(sample_stats["call_count"] * scale)
        projected_in = round(sample_stats["input_tokens"] * scale)
        projected_out = round(sample_stats["output_tokens"] * scale)
        projected_images = round(sample_stats["images_processed"] * scale)
        projected_seconds = round(sample_stats["elapsed_seconds"] * scale)
        projected_cost = round(
            (projected_in / 1000) * cost_in_per_1k + (projected_out / 1000) * cost_out_per_1k, 4
        )
        lines.append("### Test set processing (projected from sample, scaled "
                      f"x{scale:.2f} for {test_row_count} rows)")
        lines.append(f"- Projected model calls: {projected_calls}")
        lines.append(f"- Projected images processed: {projected_images}")
        lines.append(f"- Projected input tokens: {projected_in}")
        lines.append(f"- Projected output tokens: {projected_out}")
        lines.append(f"- Projected runtime: ~{projected_seconds}s "
                      f"(serialized; lower with batching/parallelism)")
        lines.append(f"- Projected cost: ${projected_cost}\n")
    else:
        lines.append("### Test set processing\n(no stats available — run "
                      "run_pipeline.py against the test set to populate this)\n")

    lines.append("### Rate limits, batching, and reliability strategy")
    lines.append(
        "- One Gemini call per claim row (1 claim = 1 call, all of that "
        "claim's images attached together) — this keeps reasoning grounded "
        "across all images for a claim at once, instead of fragmenting "
        "context per image.\n"
        "- Calls are paced with a fixed minimum interval "
        f"({config.MIN_SECONDS_BETWEEN_CALLS}s) between requests to stay "
        "under free-tier RPM limits.\n"
        "- Transient/rate-limit failures are retried with exponential "
        f"backoff, up to {config.MAX_RETRIES} attempts, before falling back "
        "to a safe default (not_enough_information + manual_review_required) "
        "rather than crashing the run.\n"
        "- Claims with no loadable images skip the API call entirely "
        "(no point spending a call on unusable input) and are marked "
        "not_enough_information directly.\n"
        "- No response caching is implemented since claims are processed "
        "once each; if reruns become common, caching by "
        "(user_claim, image_paths) hash would avoid redundant calls."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--sample-stats", default=None)
    parser.add_argument("--test-stats", default=None)
    parser.add_argument("--test-row-count", type=int, default=None,
                         help="Number of rows in the real test set, for projecting cost")
    parser.add_argument("--cost-per-1k-input-tokens", type=float, default=0.0)
    parser.add_argument("--cost-per-1k-output-tokens", type=float, default=0.0)
    parser.add_argument("--out", default="evaluation/evaluation_report.md")
    parser.add_argument(
        "--compare", action="append", default=[], metavar="LABEL:CSV:STATS_JSON",
        help="Add a strategy to compare, format 'label:predicted.csv:stats.json' "
             "(stats.json optional, use 'label:predicted.csv' to omit). Repeat "
             "this flag once per strategy, e.g. "
             "--compare \"flash:flash_output.csv:flash_stats.json\" "
             "--compare \"flash-lite:lite_output.csv:lite_stats.json\"",
    )
    args = parser.parse_args()

    gold_rows = load_csv(args.gold)
    pred_rows = load_csv(args.predicted)
    accuracy_report = evaluate(gold_rows, pred_rows)

    sample_stats = load_stats(args.sample_stats)
    test_stats = load_stats(args.test_stats)

    comparison_results = None
    if args.compare:
        strategy_predictions = []
        for spec in args.compare:
            parts = spec.split(":")
            label = parts[0]
            csv_path = parts[1]
            stats_path = parts[2] if len(parts) > 2 else None
            strategy_predictions.append(
                (label, load_csv(csv_path), load_stats(stats_path))
            )
        comparison_results = compare_strategies(gold_rows, strategy_predictions)

    write_report(
        args.out, accuracy_report, sample_stats, test_stats, args.test_row_count,
        args.cost_per_1k_input_tokens, args.cost_per_1k_output_tokens,
        comparison_results=comparison_results,
    )
    print(json.dumps(accuracy_report, indent=2))


if __name__ == "__main__":
    main()
