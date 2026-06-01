#!/usr/bin/env python3
"""
Download the last 100 pending source images from sortitoutsi.net queue.
Saves files to ./input/ using the image's alt name.

Usage:
    SITSI_COOKIE="your_session_cookie" python3 download_queue.py

To get your session cookie:
    1. Open sortitoutsi.net in your browser and log in
    2. Open DevTools (F12) → Application → Cookies → sortitoutsi.net
    3. Copy the value of the session/login cookie (e.g. 'laravel_session' or 'remember_web_...')
    4. Run: SITSI_COOKIE="name=value; name2=value2" python3 download_queue.py
"""

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://sortitoutsi.net"
QUEUE_URL = (
    "https://sortitoutsi.net/graphics/submissions/1/queue"
    "?type=source&status=pending&megapack_status=&inpack=&new_player="
    "&game_item_id=&submitted_by_id=&sort=submitted_at-desc&submit=1"
)
INPUT_DIR = Path(__file__).parent / "input"
MAX_IMAGES = 50


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
    # Rebuild query string preserving order
    new_query = "&".join(f"{k}={v[0]}" for k, v in params.items())
    return urlunparse(parsed._replace(query=new_query))


def safe_filename(name: str) -> str:
    """Turn an alt-name into a safe filename, keeping the extension if any."""
    name = name.strip()
    # Replace path separators and other unsafe chars
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name


def collect_image_entries(session: requests.Session) -> list[dict]:
    """Scrape queue pages and collect up to MAX_IMAGES image entries."""
    entries = []
    page = 1
    seen_urls = set()

    while len(entries) < MAX_IMAGES:
        url = QUEUE_URL + f"&page={page}"
        print(f"Fetching page {page}…")
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} — stopping.", file=sys.stderr)
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Images are shown as <img> tags with a src pointing to the CDN
        # They live inside submission cards/rows
        imgs = soup.select("img[src*='/graphics/']")
        if not imgs:
            # Try broader selector
            imgs = soup.select("img[alt]")

        found_on_page = 0
        for img in imgs:
            src = img.get("src", "")
            alt = img.get("alt", "").strip()
            if not src or not alt:
                continue
            if src in seen_urls:
                continue
            # Skip tiny UI icons (width/height attrs ≤ 30px heuristic)
            w = img.get("width", "")
            h = img.get("height", "")
            try:
                if int(w) <= 30 or int(h) <= 30:
                    continue
            except (ValueError, TypeError):
                pass

            full_url = src if src.startswith("http") else BASE_URL + src
            full_url = strip_size_params(full_url)

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


def download_images(session: requests.Session, entries: list[dict]):
    INPUT_DIR.mkdir(exist_ok=True)
    total = len(entries)
    ok = 0
    for i, entry in enumerate(entries, 1):
        url = entry["url"]
        alt = entry["alt"]
        print(f"[{i}/{total}] {alt}")
        try:
            resp = session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            ext = guess_extension(url, resp.headers.get("Content-Type", ""))
            # Use alt name as filename; add extension if not already present
            base = safe_filename(alt)
            if not base.lower().endswith(ext.lower()):
                filename = base + ext
            else:
                filename = base
            dest = INPUT_DIR / filename
            # Avoid overwriting; append index if needed
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


def main():
    cookie_str = os.environ.get("SITSI_COOKIE", "")
    if not cookie_str:
        print(
            "ERROR: Set the SITSI_COOKIE environment variable with your browser cookies.\n"
            "\n"
            "How to get it:\n"
            "  1. Log in to sortitoutsi.net in your browser\n"
            "  2. Open DevTools → Application → Cookies → sortitoutsi.net\n"
            "  3. Copy all cookie name=value pairs, separated by semicolons\n"
            "  4. Run: SITSI_COOKIE='laravel_session=abc123; remember_web_...=xyz' python3 download_queue.py",
            file=sys.stderr,
        )
        sys.exit(1)

    session = get_session(cookie_str)
    entries = collect_image_entries(session)
    if not entries:
        print(
            "No images found. Check your cookie / the page structure.", file=sys.stderr
        )
        sys.exit(1)

    print(f"\nCollected {len(entries)} images. Starting download…\n")
    download_images(session, entries)


if __name__ == "__main__":
    main()
