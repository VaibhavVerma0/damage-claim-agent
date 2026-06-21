"""
Central configuration: model choice + every allowed-value list from the
problem statement. Keeping these in one place means validation logic and
prompt-building both read from the same source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Gemini 2.5 Flash-Lite: most generous free-tier daily quota among current
# models (gemini-2.5-flash was cut to ~20 requests/day in Dec 2025; Flash-Lite
# is meaningfully higher). Still supports vision + forced JSON output.
# Check your actual current limits at https://aistudio.google.com/rate-limit
# since Google adjusts these per-account rather than publishing a fixed table.
MODEL_NAME = "gemini-2.5-flash-lite"

API_KEY_ENV_VAR = "GEMINI_API_KEY"

# Pacing between calls. Daily quota (RPD) resets at midnight Pacific time,
# not on a rolling 24h window — check aistudio.google.com/rate-limit for
# your exact RPM/RPD numbers and adjust this if you're hitting 429s.
MIN_SECONDS_BETWEEN_CALLS = 5.0
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 8

# ---------------------------------------------------------------------------
# Allowed output values (from the problem statement)
# ---------------------------------------------------------------------------

CLAIM_STATUS_VALUES = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE_VALUES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain",
    "none", "unknown",
]

OBJECT_PART_VALUES = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    ],
}

RISK_FLAG_VALUES = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]

SEVERITY_VALUES = ["none", "low", "medium", "high", "unknown"]

CLAIM_OBJECT_VALUES = ["car", "laptop", "package"]

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV_VAR, "")
    if not key:
        raise RuntimeError(
            f"Set the {API_KEY_ENV_VAR} environment variable "
            f"(get a free key at https://aistudio.google.com/apikey)"
        )
    return key
