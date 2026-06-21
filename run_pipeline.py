"""
Entry point. Run against the labeled sample set first to sanity-check,
then against the real test set.

Usage:
    python run_pipeline.py \
        --claims dataset/claims.csv \
        --user-history dataset/user_history.csv \
        --evidence dataset/evidence_requirements.csv \
        --images-root dataset \
        --output output.csv \
        --stats-out evaluation/run_stats_test.json
"""

import argparse
import csv
import json
import time

import config
import io_utils
from claim_agent import process_claim
from gemini_client import GeminiClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, help="Path to claims CSV (sample or test)")
    parser.add_argument(
        "--user-history", default=None,
        help="Path to user_history.csv. Optional — if omitted, history context is skipped.",
    )
    parser.add_argument(
        "--evidence", default=None,
        help="Path to evidence_requirements.csv. Optional — if omitted, no specific "
             "evidence rules are passed to the model.",
    )
    parser.add_argument(
        "--images-root", required=True,
        help="Folder that image_paths in the CSV are relative to (usually the dataset/ root)",
    )
    parser.add_argument("--output", required=True, help="Where to write output.csv")
    parser.add_argument(
        "--stats-out", default=None,
        help="Where to write run stats JSON (for the evaluation report)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N rows (useful for a quick test)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override the model from config.py (e.g. gemini-2.5-flash, "
             "gemini-2.5-flash-lite) — useful for comparing strategies.",
    )
    args = parser.parse_args()

    claims = io_utils.load_csv_rows(args.claims)
    if args.limit:
        claims = claims[: args.limit]
    user_history = io_utils.load_user_history(args.user_history) if args.user_history else {}
    evidence_by_object = (
        io_utils.load_evidence_requirements(args.evidence) if args.evidence else {}
    )

    model_name = args.model or config.MODEL_NAME
    print(f"Using model: {model_name}")
    client = GeminiClient(model_name=model_name)
    client.stats.start()

    results = []
    for i, row in enumerate(claims, start=1):
        print(f"[{i}/{len(claims)}] processing claim for user_id={row.get('user_id')}")
        result = process_claim(row, client, evidence_by_object, user_history, args.images_root)
        results.append(result)

    client.stats.stop()

    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.OUTPUT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({col: row.get(col, "") for col in config.OUTPUT_COLUMNS})

    print(f"\nWrote {len(results)} rows to {args.output}")
    print("Run stats:", json.dumps(client.stats.as_dict(), indent=2))

    if args.stats_out:
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.stats_out)) or ".", exist_ok=True)
        stats_dict = client.stats.as_dict()
        stats_dict["model"] = model_name
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Stats written to {args.stats_out}")


if __name__ == "__main__":
    main()
