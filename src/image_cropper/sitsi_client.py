"""Shared authenticated HTTP client for sortitoutsi.net.

All network calls that carry the session cookie go through here so
validation and SSRF guards are enforced in one place.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

from image_cropper.errors import ValidationError

BASE_URL: str = "https://sortitoutsi.net"
ALLOWED_HOSTS: frozenset[str] = frozenset({
    "sortitoutsi.net",
    "www.sortitoutsi.net",
    "sortitoutsi.b-cdn.net",  # CDN that serves all uploaded media
})

_COOKIE_PAIR_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+=[^;]*$")


def validate_cookie_string(raw: str) -> str:
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
    """Reject URLs not on the sortitoutsi.net domain (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(f"refusing non-HTTP(S) URL: {url}")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValidationError(f"refusing URL outside sortitoutsi.net: {url}")
    return url


def validate_sitsi_url(url: str) -> str:
    """Like validate_image_url but for any sortitoutsi page URL."""
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
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": BASE_URL,
        }
    )
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            name, _, value = part.partition("=")
            session.cookies.set(name.strip(), value.strip(), domain="sortitoutsi.net")
    return session


def get_csrf_token(session: requests.Session, page_url: str) -> str:
    """Fetch *page_url* and extract the Laravel CSRF token.

    Tries (in order):
    1. ``<meta name="csrf-token" content="...">``
    2. ``<input name="_token" value="...">``

    Raises :class:`ValidationError` if no token is found.
    """
    validate_sitsi_url(page_url)
    resp = session.get(page_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    meta = soup.find("meta", attrs={"name": "csrf-token"})
    if isinstance(meta, Tag) and meta.get("content"):
        return str(meta["content"])

    inp = soup.find("input", attrs={"name": "_token"})
    if isinstance(inp, Tag) and inp.get("value"):
        return str(inp["value"])

    raise ValidationError(f"could not find CSRF token on {page_url}")


def get_hidden_form_fields(soup: BeautifulSoup, form_selector: str = "form") -> dict[str, str]:
    """Return all hidden input values from the first matching form."""
    form = soup.select_one(form_selector)
    if not form:
        return {}
    return {
        str(inp["name"]): str(inp.get("value", ""))
        for inp in form.find_all("input", attrs={"type": "hidden"})
        if isinstance(inp, Tag) and inp.get("name")
    }
