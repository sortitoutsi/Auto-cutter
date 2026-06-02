#!/usr/bin/env python3
"""
Download pending source images from the sortitoutsi.net submission queue.
Saves files to ./input/ using each image's alt name.

Usage:
    SITSI_COOKIE="your_session_cookie" python -m image_cropper.pipeline.download_queue

To get your session cookie:
    1. Open sortitoutsi.net in your browser and log in
    2. Open DevTools (F12) → Application → Cookies → sortitoutsi.net
    3. Copy the value of the session/login cookie (e.g. 'laravel_session' or 'remember_web_...')
    4. Run with SITSI_COOKIE="name=value; name2=value2"
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from image_cropper.errors import ImageCropperError, ValidationError
from image_cropper.types import QueueEntry

BASE_URL: str = "https://sortitoutsi.net"
ALLOWED_HOSTS: set[str] = {"sortitoutsi.net", "www.sortitoutsi.net"}
QUEUE_URL: str = (
    "https://sortitoutsi.net/graphics/submissions/1/queue"
    "?type=source&status=pending&megapack_status=&inpack=&new_player="
    "&game_item_id=&submitted_by_id=&sort=submitted_at-desc&submit=1"
)
INPUT_DIR: Path = Path(__file__).parent / "input"
MAX_IMAGES: int = 50

# RFC-style cookie value: name=value pairs, separated by ";"
# Names are restricted to token chars per RFC 6265; we accept the common set.
_COOKIE_PAIR_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+=[^;]*$")


def validate_cookie_string(raw: str) -> str:
    """Validate a `name=value(;name=value)*` cookie string.

    Raises :class:`ValidationError` if the string contains no recognizable
    name=value pairs.
    """
    if not raw or not raw.strip():
        raise ValidationError("SITSI_COOKIE is empty")
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    valid = [p for p in parts if _COOKIE_PAIR_RE.match(p)]
    if not valid:
        raise ValidationError(
            f"SITSI_COOKIE has no valid name=value pairs (got {len(parts)} segment(s))"
        )
    return raw


def validate_image_url(url: str) -> str:
    """Reject URLs not on the sortitoutsi.net domain.

    Light SSRF guard since the session cookie travels with these requests.
    Raises :class:`ValidationError` for any other host or non-HTTPS scheme.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(f"refusing non-HTTP(S) URL: {url}")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValidationError(f"refusing URL outside sortitoutsi.net: {url}")
    return url


def get_session(cookie_str: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36",
            "Referer": BASE_URL,
        }
    )
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            name, _, value = part.partition("=")
            session.cookies.set(name.strip(), value.strip(), domain="sortitoutsi.net")
    return session


def strip_size_params(url: str) -> str:
    """Remove width= and height= query parameters from an image URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("width", None)
    params.pop("height", None)
    new_query = "&".join(f"{k}={v[0]}" for k, v in params.items())
    return urlunparse(parsed._replace(query=new_query))


def safe_filename(name: str) -> str:
    """Turn an alt-name into a safe filename, keeping any existing extension."""
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name


def collect_image_entries(session: requests.Session) -> list[QueueEntry]:
    """Scrape queue pages and collect up to MAX_IMAGES image entries."""
    entries: list[QueueEntry] = []
    page = 1
    seen_urls: set[str] = set()

    while len(entries) < MAX_IMAGES:
        url = QUEUE_URL + f"&page={page}"
        print(f"Fetching page {page}…")
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} — stopping.", file=sys.stderr)
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        imgs = soup.select("img[src*='/graphics/']")
        if not imgs:
            imgs = soup.select("img[alt]")

        found_on_page = 0
        for img in imgs:
            src = img.get("src", "")
            alt_raw = img.get("alt", "")
            if isinstance(src, list):
                src = src[0] if src else ""
            if isinstance(alt_raw, list):
                alt_raw = alt_raw[0] if alt_raw else ""
            alt = alt_raw.strip()
            if not src or not alt:
                continue
            if src in seen_urls:
                continue
            w = img.get("width", "")
            h = img.get("height", "")
            try:
                if int(str(w)) <= 30 or int(str(h)) <= 30:
                    continue
            except (ValueError, TypeError):
                pass

            full_url = src if src.startswith("http") else BASE_URL + src
            full_url = strip_size_params(full_url)
            try:
                validate_image_url(full_url)
            except ValidationError as e:
                print(f"  skip: {e}", file=sys.stderr)
                continue

            seen_urls.add(src)
            entries.append({"url": full_url, "alt": alt})
            found_on_page += 1

            if len(entries) >= MAX_IMAGES:
                break

        print(f"  Found {found_on_page} images (total so far: {len(entries)})")

        if found_on_page == 0:
            print("  No new images on this page — done.")
            break

        page += 1
        time.sleep(0.5)

    return entries[:MAX_IMAGES]


def guess_extension(url: str, content_type: str) -> str:
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


def download_images(session: requests.Session, entries: list[QueueEntry]) -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    total = len(entries)
    ok = 0
    for i, entry in enumerate(entries, 1):
        url = entry["url"]
        alt = entry["alt"]
        try:
            validate_image_url(url)
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
            dest = INPUT_DIR / filename
            if dest.exists():
                stem = Path(filename).stem
                dest = INPUT_DIR / f"{stem}_{i}{ext}"
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            size_kb = dest.stat().st_size // 1024
            print(f"    → {dest.name} ({size_kb} KB)")
            ok += 1
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
        time.sleep(0.2)

    print(f"\nDone. Downloaded {ok}/{total} images to {INPUT_DIR}/")


def main() -> None:
    cookie_str = os.environ.get("SITSI_COOKIE", "")
    try:
        if not cookie_str:
            raise ValidationError(
                "Set the SITSI_COOKIE environment variable with your browser cookies.\n"
                "\n"
                "How to get it:\n"
                "  1. Log in to sortitoutsi.net in your browser\n"
                "  2. Open DevTools → Application → Cookies → sortitoutsi.net\n"
                "  3. Copy all cookie name=value pairs, separated by semicolons\n"
                "  4. Run: SITSI_COOKIE='laravel_session=abc123; remember_web_...=xyz' "
                "python -m image_cropper.pipeline.download_queue"
            )
        validate_cookie_string(cookie_str)
    except ImageCropperError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    session = get_session(cookie_str)
    entries = collect_image_entries(session)
    if not entries:
        print("No images found. Check your cookie / the page structure.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCollected {len(entries)} images. Starting download…\n")
    download_images(session, entries)


if __name__ == "__main__":
    main()
