#!/usr/bin/env python3
"""
Download source images from sortitoutsi.net.

Two modes:
  Queue mode (default) — pulls from the pending-submissions queue.
  Collection mode      — pulls from a specific collection page.

In collection mode a ``<stem>.sitsi.json`` sidecar is written next to each
image so the submit step can post the finished cutout back.

Usage:
    # Queue mode
    SITSI_COOKIE="..." python -m image_cropper.pipeline.download_queue

    # Collection mode
    SITSI_COOKIE="..." python -m image_cropper.pipeline.download_queue \\
        --collection https://sortitoutsi.net/graphics/submissions/collection/59843

To get your session cookie:
    1. Log in to sortitoutsi.net in your browser
    2. Open DevTools → Application → Cookies → sortitoutsi.net
    3. Copy all name=value pairs separated by semicolons
    4. Export as SITSI_COOKIE="laravel_session=abc123; remember_web_...=xyz"
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

from image_cropper.errors import ImageCropperError, ValidationError
from image_cropper.sitsi_client import (
    BASE_URL,
    get_session,
    validate_cookie_string,
    validate_url,
)
from image_cropper.types import QueueEntry, SubmissionMeta

INPUT_DIR: Path = Path(os.environ.get("SITSI_INPUT_DIR", str(Path.home() / "image-cropper-output" / "input")))
MAX_IMAGES: int = 50

QUEUE_URL: str = (
    "https://sortitoutsi.net/graphics/submissions/1/queue"
    "?type=source&status=pending&megapack_status=&inpack=&new_player="
    "&game_item_id=&submitted_by_id=&sort=submitted_at-desc&submit=1"
)

DOWNLOADABLE_STATUSES: frozenset[str] = frozenset({"pending", "in_progress"})


# ── URL helpers ────────────────────────────────────────────────────────────────

def strip_size_params(url: str) -> str:
    """Remove ``width`` and ``height`` query parameters from *url*.

    sortitoutsi.net serves thumbnails via ``?width=N&height=N``.  Stripping
    those parameters yields the original full-resolution image URL.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("width", None)
    params.pop("height", None)
    new_query = "&".join(f"{k}={v[0]}" for k, v in params.items())
    return urlunparse(parsed._replace(query=new_query))


def safe_filename(name: str) -> str:
    """Sanitise *name* for use as a filesystem filename.

    Strips leading/trailing whitespace and replaces characters that are
    illegal on at least one major OS (the Windows set: ``\\/:*?"<>|``)
    with underscores.
    """
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name


def guess_extension(url: str, content_type: str) -> str:
    """Infer a file extension from the URL path or HTTP Content-Type header.

    The URL path suffix is checked first; if it has none the Content-Type is
    consulted.  Defaults to ``'.jpg'`` when neither source is conclusive.
    """
    path = urlparse(url).path
    ext = Path(path).suffix
    if ext:
        return ext
    ct = content_type.split(";")[0].strip()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(ct, ".jpg")


# ── Queue mode ─────────────────────────────────────────────────────────────────

def collect_image_entries(
    session: requests.Session,
    queue_url: str = QUEUE_URL,
) -> list[tuple[QueueEntry, SubmissionMeta]]:
    """Scrape queue pages and return up to MAX_IMAGES (entry, metadata) pairs.

    Each submission is identified by the ``a.lightgallery-link`` anchor whose
    ``href`` is the full-resolution CDN image URL.  Submission ID and person ID
    are extracted from sibling links in the same container row.

    Returns the same ``list[tuple[QueueEntry, SubmissionMeta]]`` shape as
    ``collect_collection_entries`` so callers can treat both modes uniformly.
    """
    results: list[tuple[QueueEntry, SubmissionMeta]] = []
    page = 1
    seen_urls: set[str] = set()
    base_url = queue_url.rstrip("&")

    while len(results) < MAX_IMAGES:
        paged_url = f"{base_url}&page={page}"
        print(f"Fetching queue page {page}…")
        resp = session.get(paged_url, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} — stopping.", file=sys.stderr)
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        found_on_page = 0

        for link in soup.select("a.lightgallery-link"):
            img_url = str(link.get("href", "")).strip()
            if not img_url or img_url in seen_urls:
                continue

            try:
                validate_url(img_url)
            except ValidationError as e:
                print(f"  skip: {e}", file=sys.stderr)
                continue

            # Walk up to the flex-md-row container that wraps one submission
            container = link.find_parent("div", class_="flex-md-row")
            if container is None:
                continue

            # Submission ID from /submissions/show/<id>
            submission_id: int | None = None
            sub_link = container.find("a", href=re.compile(r"/submissions/show/\d+"))
            if isinstance(sub_link, Tag):
                m = re.search(r"/submissions/show/(\d+)", str(sub_link.get("href", "")))
                if m:
                    submission_id = int(m.group(1))
            if submission_id is None:
                continue  # cannot track submission without ID

            # Person ID from /browse/<megapack>/<person_id>/
            person_id: int | None = None
            person_link = container.find("a", href=re.compile(r"/browse/\d+/\d+"))
            if isinstance(person_link, Tag):
                m2 = re.search(r"/browse/\d+/(\d+)", str(person_link.get("href", "")))
                if m2:
                    person_id = int(m2.group(1))

            # Player name from the gameitemname span
            name_tag = container.find("span", class_="gameitemname")
            alt = name_tag.get_text(strip=True) if isinstance(name_tag, Tag) else f"submission_{submission_id}"

            seen_urls.add(img_url)
            entry: QueueEntry = {"url": img_url, "alt": alt}
            meta: SubmissionMeta = {
                "submission_id": submission_id,
                "person_id": person_id,
                "alt": alt,
                "status": "pending",
                "image_type": "source",
                "collection_url": queue_url,
                "downloaded_at": datetime.now(tz=UTC).isoformat(),
            }
            results.append((entry, meta))
            found_on_page += 1
            if len(results) >= MAX_IMAGES:
                break

        print(f"  Found {found_on_page} images on page {page} (total: {len(results)})")
        if found_on_page == 0:
            print("  No new images on this page — done.")
            break
        page += 1
        time.sleep(0.5)

    return results[:MAX_IMAGES]


# ── Collection mode ────────────────────────────────────────────────────────────

def _parse_submission_id_from_url(href: str) -> int | None:
    m = re.search(r"/submissions(?:/[^/]+)?/(\d+)", href)
    return int(m.group(1)) if m else None


def _parse_person_id_from_url(href: str) -> int | None:
    m = re.search(r"/browse/\d+/(\d+)", href)
    return int(m.group(1)) if m else None


def _parse_status(row_html: str) -> str:
    lower = row_html.lower()
    for status in ("completed", "rejected", "in_progress", "in-progress", "pending"):
        if status.replace("-", "_") in lower or status in lower:
            return status.replace("-", "_")
    return "pending"


def _is_game_ready(row_html: str) -> bool:
    return "game_ready" in row_html.lower() or "game-ready" in row_html.lower()


def collect_collection_entries(
    session: requests.Session, collection_url: str
) -> list[tuple[QueueEntry, SubmissionMeta]]:
    """Scrape a collection page; return (image_entry, metadata) for downloadable sources."""
    results: list[tuple[QueueEntry, SubmissionMeta]] = []
    page = 1

    while len(results) < MAX_IMAGES:
        paged_url = collection_url if page == 1 else f"{collection_url}?page={page}"
        print(f"Fetching collection page {page}…")
        resp = session.get(paged_url, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} — stopping.", file=sys.stderr)
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Try common sortitoutsi row selectors, fall back to submission links
        rows = (
            soup.select(".submission-row")
            or soup.select("tr[data-submission-id]")
            or soup.select("[data-id]")
            or soup.select(".submission")
            or soup.find_all("a", href=re.compile(r"/submissions/\d+"))
        )

        found_on_page = 0
        for row in rows:
            row_html = str(row)

            # --- submission ID ---
            submission_id: int | None = None
            for attr in ("data-submission-id", "data-id"):
                val = row.get(attr)
                if val:
                    with contextlib.suppress(ValueError):
                        submission_id = int(str(val))
                    break
            if submission_id is None:
                href = row.get("href", "") if row.name == "a" else ""
                submission_id = _parse_submission_id_from_url(str(href))
                if submission_id is None:
                    link = row.find("a", href=re.compile(r"/submissions/\d+"))
                    if isinstance(link, Tag):
                        submission_id = _parse_submission_id_from_url(str(link["href"]))
            if submission_id is None:
                continue

            # --- status + type ---
            status = _parse_status(row_html)
            game_ready = _is_game_ready(row_html)

            if status not in DOWNLOADABLE_STATUSES or game_ready:
                print(
                    f"  skip {submission_id}: status={status} game_ready={game_ready}"
                )
                continue

            # --- person ID ---
            person_id: int | None = None
            person_link = row.find("a", href=re.compile(r"/browse/\d+/\d+"))
            if isinstance(person_link, Tag):
                person_id = _parse_person_id_from_url(str(person_link["href"]))

            # --- source image ---
            img = row.find("img")
            if not isinstance(img, Tag):
                continue
            src = str(img.get("src", ""))
            alt = str(img.get("alt", f"submission_{submission_id}")).strip()
            if not src:
                continue

            full_url = src if src.startswith("http") else BASE_URL + src
            full_url = strip_size_params(full_url)
            try:
                validate_url(full_url)
            except ValidationError as e:
                print(f"  skip: {e}", file=sys.stderr)
                continue

            entry: QueueEntry = {"url": full_url, "alt": alt}
            meta: SubmissionMeta = {
                "submission_id": submission_id,
                "person_id": person_id,
                "alt": alt,
                "status": status,
                "image_type": "source",
                "collection_url": collection_url,
                "downloaded_at": datetime.now(tz=UTC).isoformat(),
            }
            results.append((entry, meta))
            found_on_page += 1
            if len(results) >= MAX_IMAGES:
                break

        print(
            f"  Found {found_on_page} downloadable source images (total: {len(results)})"
        )
        if found_on_page == 0:
            break
        page += 1
        time.sleep(0.5)

    return results[:MAX_IMAGES]


# ── Download ───────────────────────────────────────────────────────────────────

def download_images(
    session: requests.Session,
    entries: list[QueueEntry],
    out_dir: Path,
    metas: list[SubmissionMeta] | None = None,
) -> None:
    """Download *entries* to *out_dir*, optionally writing sidecar metadata files.

    Images are streamed in 8 KB chunks.  If the destination filename already
    exists an index suffix is appended to avoid overwriting.

    When *metas* is provided (collection mode), a ``<stem>.sitsi.json``
    sidecar is written next to each image.  The sidecar is later read by
    ``submit_cutout`` so the finished cutout can be posted back without the
    user re-entering the submission ID.

    A 200 ms pause between requests avoids hammering the server.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    ok = 0
    for i, entry in enumerate(entries, 1):
        url = entry["url"]
        alt = entry["alt"]
        try:
            validate_url(url)
        except ValidationError as e:
            print(f"    SKIP: {e}", file=sys.stderr)
            continue

        print(f"[{i}/{total}] {alt}")
        try:
            resp = session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            ext = guess_extension(url, resp.headers.get("Content-Type", ""))
            base = safe_filename(alt)
            filename = base if base.lower().endswith(ext.lower()) else base + ext
            dest = out_dir / filename
            if dest.exists():
                dest = out_dir / f"{Path(filename).stem}_{i}{ext}"
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            size_kb = dest.stat().st_size // 1024
            print(f"    → {dest.name} ({size_kb} KB)")

            if metas is not None:
                sidecar = dest.with_suffix(".sitsi.json")
                sidecar.write_text(json.dumps(metas[i - 1], indent=2), encoding="utf-8")
                print(f"    → {sidecar.name}")

            ok += 1
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
        time.sleep(0.2)

    print(f"\nDone. Downloaded {ok}/{total} images to {out_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download source images from sortitoutsi.net"
    )
    parser.add_argument(
        "--collection",
        metavar="URL",
        help="Download from a specific collection URL instead of the pending queue",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=str(INPUT_DIR),
        help="Output directory (default: input/)",
    )
    args = parser.parse_args()

    # Cookie is required for collection mode (authenticated endpoints) but
    # optional for queue mode (the queue page is publicly accessible).
    cookie_str = os.environ.get("SITSI_COOKIE", "")
    if cookie_str:
        try:
            validate_cookie_string(cookie_str)
        except ImageCropperError as e:
            print(f"ERROR: invalid SITSI_COOKIE — {e}", file=sys.stderr)
            sys.exit(1)
    elif args.collection:
        print(
            "ERROR: SITSI_COOKIE is required for collection mode.\n"
            "  1. Log in to sortitoutsi.net in your browser\n"
            "  2. Open DevTools → Application → Cookies → sortitoutsi.net\n"
            "  3. Copy all cookie name=value pairs, separated by semicolons\n"
            "  4. Run: SITSI_COOKIE='laravel_session=abc123' "
            "python -m image_cropper.pipeline.download_queue --collection URL",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output)
    session = get_session(cookie_str)

    if args.collection:
        pairs = collect_collection_entries(session, args.collection)
        if not pairs:
            print(
                "No downloadable source images found in the collection. "
                "Check your cookie or the collection URL.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        pairs = collect_image_entries(session)
        if not pairs:
            print("No images found in the queue.", file=sys.stderr)
            sys.exit(1)

    image_entries = [p[0] for p in pairs]
    metas = [p[1] for p in pairs]
    print(f"\nCollected {len(image_entries)} source images. Starting download…\n")
    download_images(session, image_entries, out_dir, metas)


if __name__ == "__main__":
    main()
