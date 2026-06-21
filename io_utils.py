"""
Loading helpers for the dataset: CSVs, images, and the supporting lookup
tables (user history, evidence requirements).
"""

import base64
import csv
import mimetypes
import os
from collections import defaultdict


def load_csv_rows(path):
    """Load a CSV into a list of dicts (preserves column order via DictReader)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_user_history(path):
    """Returns {user_id: {...row...}}"""
    rows = load_csv_rows(path)
    return {row["user_id"]: row for row in rows}


def load_evidence_requirements(path):
    """
    Returns {claim_object: [list of requirement rows]}, where rows tagged
    'all' are duplicated into every object's list (since they apply
    universally per the schema).
    """
    rows = load_csv_rows(path)
    by_object = defaultdict(list)
    universal = []
    for row in rows:
        obj = row.get("claim_object", "").strip().lower()
        if obj == "all":
            universal.append(row)
        else:
            by_object[obj].append(row)
    for obj in list(by_object.keys()):
        by_object[obj] = by_object[obj] + universal
    # also expose universal-only fallback for any object not explicitly listed
    by_object["_universal"] = universal
    return by_object


def parse_image_paths(image_paths_field):
    """'images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg' -> list"""
    if not image_paths_field:
        return []
    return [p.strip() for p in image_paths_field.split(";") if p.strip()]


def image_id_from_path(path):
    """images/test/case_001/img_1.jpg -> img_1"""
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]


def load_image_as_base64(full_path):
    """Returns (mime_type, base64_string). Raises if the file is missing."""
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    mime_type, _ = mimetypes.guess_type(full_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(full_path, "rb") as f:
        data = f.read()
    return mime_type, base64.b64encode(data).decode("utf-8")
