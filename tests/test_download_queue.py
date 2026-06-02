"""Tests for the pure helpers in `pipeline.download_queue`.

No network access; only the deterministic string/URL helpers are exercised.
"""

from __future__ import annotations

import pytest

from image_cropper.errors import ValidationError
from image_cropper.pipeline.download_queue import (
    guess_extension,
    safe_filename,
    strip_size_params,
    validate_cookie_string,
    validate_image_url,
)

# ---------------------------------------------------------------------------
# strip_size_params
# ---------------------------------------------------------------------------


def test_strip_size_params_removes_width_height() -> None:
    url = "https://example.com/img.png?width=100&height=200&id=42"
    out = strip_size_params(url)
    assert "width=" not in out
    assert "height=" not in out
    assert "id=42" in out


def test_strip_size_params_no_query_unchanged() -> None:
    url = "https://example.com/img.png"
    assert strip_size_params(url) == url


# ---------------------------------------------------------------------------
# safe_filename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hello.png", "hello.png"),
        ("hello / world", "hello _ world"),
        ('path\\name:with*bad?chars"<>|', "path_name_with_bad_chars____"),
        ("  trim  ", "trim"),
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


# ---------------------------------------------------------------------------
# guess_extension
# ---------------------------------------------------------------------------


def test_guess_extension_uses_path_suffix() -> None:
    assert guess_extension("https://x.com/a.webp", "image/png") == ".webp"


def test_guess_extension_falls_back_to_content_type() -> None:
    assert guess_extension("https://x.com/a", "image/png; charset=binary") == ".png"


def test_guess_extension_unknown_defaults_to_jpg() -> None:
    assert guess_extension("https://x.com/a", "application/octet-stream") == ".jpg"


# ---------------------------------------------------------------------------
# validate_cookie_string
# ---------------------------------------------------------------------------


def test_validate_cookie_string_accepts_single_pair() -> None:
    assert validate_cookie_string("session=abc123") == "session=abc123"


def test_validate_cookie_string_accepts_multiple_pairs() -> None:
    assert validate_cookie_string("a=1; b=2; c=3") == "a=1; b=2; c=3"


def test_validate_cookie_string_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_cookie_string("")


def test_validate_cookie_string_rejects_whitespace_only() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_cookie_string("   ")


def test_validate_cookie_string_rejects_garbage() -> None:
    with pytest.raises(ValidationError, match="no valid"):
        validate_cookie_string("not a cookie at all")


# ---------------------------------------------------------------------------
# validate_image_url
# ---------------------------------------------------------------------------


def test_validate_image_url_accepts_sortitoutsi() -> None:
    url = "https://sortitoutsi.net/graphics/a.png"
    assert validate_image_url(url) == url


def test_validate_image_url_accepts_www_subdomain() -> None:
    url = "https://www.sortitoutsi.net/graphics/a.png"
    assert validate_image_url(url) == url


def test_validate_image_url_rejects_other_host() -> None:
    with pytest.raises(ValidationError, match="outside sortitoutsi.net"):
        validate_image_url("https://attacker.example/steal?cookie=1")


def test_validate_image_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ValidationError, match="non-HTTP"):
        validate_image_url("file:///etc/passwd")


def test_validate_image_url_rejects_data_uri() -> None:
    with pytest.raises(ValidationError, match="non-HTTP"):
        validate_image_url("data:text/plain,abc")
