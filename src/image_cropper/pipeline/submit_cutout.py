#!/usr/bin/env python3
"""
Submit a processed cutout back to sortitoutsi.net as a response to a source submission.

Usage:
    SITSI_COOKIE="..." python -m image_cropper.pipeline.submit_cutout cutout.png
    SITSI_COOKIE="..." python -m image_cropper.pipeline.submit_cutout cutout.png --submission-id 3825894

If --submission-id is omitted, the script reads the <stem>.sitsi.json sidecar
written by download_queue.py in collection mode.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from image_cropper.errors import ImageCropperError, ValidationError
from image_cropper.sitsi_client import (
    BASE_URL,
    get_session,
    validate_cookie_string,
    validate_sitsi_url,
)
from image_cropper.types import SubmissionMeta, SubmitResult

SUBMIT_BASE_URL: str = "https://sortitoutsi.net/graphics/submissions/create/1"
TIMELINE_URL_TPL: str = "https://sortitoutsi.net/graphics/browse/1/{person_id}/timeline"
SIDECAR_SUFFIX: str = ".sitsi.json"


def load_metadata(image_path: Path) -> SubmissionMeta | None:
    """Read the .sitsi.json sidecar next to *image_path*, or return None."""
    # Walk up through possible intermediate dirs looking for a sidecar
    # keyed to the original stem (before pipeline renames like _final, _dg)
    candidates = [
        image_path.with_suffix(SIDECAR_SUFFIX),
        image_path.with_name(image_path.stem.removesuffix("_final") + SIDECAR_SUFFIX),
        image_path.with_name(image_path.stem.removesuffix("_dg") + SIDECAR_SUFFIX),
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return SubmissionMeta(**{k: data[k] for k in SubmissionMeta.__annotations__ if k in data})  # type: ignore[misc]
            except Exception:
                return None
    return None


def _find_sidecar(image_path: Path) -> Path | None:
    """Search for a .sitsi.json sidecar, including stem-prefix matches."""
    stem = image_path.stem
    # strip common pipeline suffixes
    for suffix in ("_final", "_dg", "_centered"):
        stem = stem.removesuffix(suffix)

    for parent in [image_path.parent] + list(image_path.parents):
        candidate = parent / (stem + SIDECAR_SUFFIX)
        if candidate.exists():
            return candidate
    return None


def submit_cutout(
    session: requests.Session,
    image_path: Path,
    submission_id: int,
) -> SubmitResult:
    """Post *image_path* as a cutout response to *submission_id*.

    Strategy:
    1. GET the create form with ?response_to_id=<id> to discover all hidden
       fields (CSRF token, etc.) dynamically — no hardcoded field names.
    2. POST multipart/form-data with the image + all discovered hidden fields.
    3. Parse the response to determine success.
    """
    if not image_path.exists():
        return SubmitResult(ok=False, submission_url=None, message=f"file not found: {image_path}")

    form_url = f"{SUBMIT_BASE_URL}?response_to_id={submission_id}"
    validate_sitsi_url(form_url)

    # --- Step 1: fetch the form ---
    try:
        form_resp = session.get(form_url, timeout=30)
        form_resp.raise_for_status()
    except Exception as e:
        return SubmitResult(ok=False, submission_url=None, message=f"could not fetch form: {e}")

    soup = BeautifulSoup(form_resp.text, "html.parser")

    # Find the form that contains a file input (the submission form)
    form = None
    for f in soup.find_all("form"):
        if f.find("input", attrs={"type": "file"}):
            form = f
            break
    if form is None:
        form = soup.find("form")

    hidden_fields: dict[str, str] = {}
    if form:
        for inp in form.find_all("input", attrs={"type": "hidden"}):
            name = inp.get("name")
            value = inp.get("value", "")
            if name:
                hidden_fields[str(name)] = str(value)

    # Detect the file input field name (commonly "file", "image", "cutout")
    file_field_name = "file"
    if form:
        file_inp = form.find("input", attrs={"type": "file"})
        if file_inp and file_inp.get("name"):
            file_field_name = str(file_inp["name"])

    # Detect the form action URL
    post_url = SUBMIT_BASE_URL
    if form and form.get("action"):
        action = str(form["action"])
        post_url = action if action.startswith("http") else BASE_URL + action

    validate_sitsi_url(post_url)

    # --- Step 2: POST ---
    try:
        with open(image_path, "rb") as img_f:
            files = {file_field_name: (image_path.name, img_f, "image/png")}
            # response_to_id goes as a regular field too
            data = {**hidden_fields, "response_to_id": str(submission_id)}
            post_resp = session.post(post_url, data=data, files=files, timeout=60)
    except Exception as e:
        return SubmitResult(ok=False, submission_url=None, message=f"upload failed: {e}")

    # --- Step 3: determine success ---
    # sortitoutsi typically redirects (302) to the new submission on success
    final_url = post_resp.url
    if post_resp.status_code in (200, 201, 302):
        # Check for error indicators in the response body
        body = post_resp.text.lower()
        error_indicators = ("error", "invalid", "failed", "unauthorized", "unauthenticated")
        success_indicators = ("submission", "thank", "success", "created")
        has_error = any(w in body for w in error_indicators)
        has_success = any(w in body for w in success_indicators) or post_resp.status_code in (201, 302)
        if has_success and not has_error:
            return SubmitResult(
                ok=True,
                submission_url=final_url if "submissions" in final_url else None,
                message=f"submitted successfully (HTTP {post_resp.status_code})",
            )

    return SubmitResult(
        ok=False,
        submission_url=None,
        message=f"unexpected response HTTP {post_resp.status_code}",
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit a processed cutout to sortitoutsi.net"
    )
    parser.add_argument("image", help="Path to the finished cutout PNG")
    parser.add_argument(
        "--submission-id",
        type=int,
        metavar="N",
        help="Submission ID to respond to (reads .sitsi.json sidecar if omitted)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    submission_id: int | None = args.submission_id

    if submission_id is None:
        meta = load_metadata(image_path)
        if meta is None:
            print(
                "ERROR: no .sitsi.json sidecar found and --submission-id not provided.",
                file=sys.stderr,
            )
            sys.exit(1)
        submission_id = meta["submission_id"]
        print(f"Using submission ID {submission_id} from sidecar ({meta['alt']})")

    cookie_str = os.environ.get("SITSI_COOKIE", "")
    try:
        if not cookie_str:
            raise ValidationError(
                "Set SITSI_COOKIE to your sortitoutsi.net session cookies."
            )
        validate_cookie_string(cookie_str)
    except ImageCropperError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    session = get_session(cookie_str)
    result = submit_cutout(session, image_path, submission_id)

    if result["ok"]:
        print(f"Success: {result['message']}")
        if result["submission_url"]:
            print(f"Submission URL: {result['submission_url']}")
    else:
        print(f"ERROR: {result['message']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
