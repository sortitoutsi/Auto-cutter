"""Unit tests for sitsi_client.py.

All tests are offline; no real HTTP calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from image_cropper.errors import ValidationError
from image_cropper.sitsi_client import (
    ALLOWED_HOSTS,
    BASE_URL,
    get_csrf_token,
    get_hidden_form_fields,
    get_session,
    validate_cookie_string,
    validate_image_url,
    validate_sitsi_url,
)


# ── validate_cookie_string ────────────────────────────────────────────────────


def test_cookie_string_single_pair() -> None:
    assert validate_cookie_string("session=abc123") == "session=abc123"


def test_cookie_string_multiple_pairs() -> None:
    raw = "laravel_session=abc; remember_web=xyz"
    assert validate_cookie_string(raw) == raw


def test_cookie_string_empty_raises() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_cookie_string("")


def test_cookie_string_whitespace_only_raises() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_cookie_string("   ")


def test_cookie_string_no_valid_pairs_raises() -> None:
    with pytest.raises(ValidationError, match="no valid name=value"):
        validate_cookie_string("not a cookie at all")


# ── validate_image_url / validate_sitsi_url ───────────────────────────────────


@pytest.mark.parametrize("fn", [validate_image_url, validate_sitsi_url])
def test_valid_sortitoutsi_url(fn) -> None:
    url = "https://sortitoutsi.net/graphics/submissions/1/queue"
    assert fn(url) == url


@pytest.mark.parametrize("fn", [validate_image_url, validate_sitsi_url])
def test_valid_cdn_url(fn) -> None:
    url = "https://sortitoutsi.b-cdn.net/uploads/face/source/123/photo.jpg"
    assert fn(url) == url


@pytest.mark.parametrize("fn", [validate_image_url, validate_sitsi_url])
def test_non_https_url_raises(fn) -> None:
    with pytest.raises(ValidationError, match="non-HTTP"):
        fn("ftp://sortitoutsi.net/file.jpg")


@pytest.mark.parametrize("fn", [validate_image_url, validate_sitsi_url])
def test_external_domain_raises(fn) -> None:
    with pytest.raises(ValidationError, match="outside sortitoutsi"):
        fn("https://attacker.example/steal?cookie=1")


@pytest.mark.parametrize("fn", [validate_image_url, validate_sitsi_url])
def test_file_scheme_raises(fn) -> None:
    with pytest.raises(ValidationError):
        fn("file:///etc/passwd")


# ── get_session ───────────────────────────────────────────────────────────────


def test_get_session_sets_cookies() -> None:
    session = get_session("laravel_session=abc123; remember_web=xyz")
    assert session.cookies.get("laravel_session", domain="sortitoutsi.net") == "abc123"
    assert session.cookies.get("remember_web", domain="sortitoutsi.net") == "xyz"


def test_get_session_sets_user_agent() -> None:
    session = get_session("s=x")
    assert "Mozilla" in session.headers["User-Agent"]


def test_get_session_sets_referer() -> None:
    session = get_session("s=x")
    assert session.headers["Referer"] == BASE_URL


def test_get_session_ignores_invalid_cookie_parts() -> None:
    session = get_session("valid=1; no-equals-sign; also=2")
    assert session.cookies.get("valid", domain="sortitoutsi.net") == "1"
    assert session.cookies.get("also", domain="sortitoutsi.net") == "2"


# ── get_csrf_token ────────────────────────────────────────────────────────────


def _mock_session(html: str):
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = resp
    return session


def test_get_csrf_token_from_meta_tag() -> None:
    html = '<html><head><meta name="csrf-token" content="token123"></head></html>'
    session = _mock_session(html)
    token = get_csrf_token(session, "https://sortitoutsi.net/page")
    assert token == "token123"


def test_get_csrf_token_from_input_tag() -> None:
    html = '<html><body><form><input name="_token" value="input_token_456"></form></body></html>'
    session = _mock_session(html)
    token = get_csrf_token(session, "https://sortitoutsi.net/page")
    assert token == "input_token_456"


def test_get_csrf_token_prefers_meta_over_input() -> None:
    html = (
        '<html><head><meta name="csrf-token" content="meta_token"></head>'
        '<body><form><input name="_token" value="input_token"></form></body></html>'
    )
    session = _mock_session(html)
    token = get_csrf_token(session, "https://sortitoutsi.net/page")
    assert token == "meta_token"


def test_get_csrf_token_raises_when_not_found() -> None:
    html = "<html><body><p>No token here</p></body></html>"
    session = _mock_session(html)
    with pytest.raises(ValidationError, match="CSRF token"):
        get_csrf_token(session, "https://sortitoutsi.net/page")


def test_get_csrf_token_rejects_external_url() -> None:
    session = MagicMock()
    with pytest.raises(ValidationError, match="outside sortitoutsi"):
        get_csrf_token(session, "https://evil.example/csrf")


# ── get_hidden_form_fields ────────────────────────────────────────────────────


def test_get_hidden_form_fields_extracts_fields() -> None:
    html = """
    <form>
        <input type="hidden" name="_token" value="abc">
        <input type="hidden" name="user_id" value="42">
        <input type="text" name="visible" value="skip me">
    </form>
    """
    soup = BeautifulSoup(html, "html.parser")
    fields = get_hidden_form_fields(soup)
    assert fields == {"_token": "abc", "user_id": "42"}


def test_get_hidden_form_fields_empty_when_no_form() -> None:
    soup = BeautifulSoup("<html><body>no form</body></html>", "html.parser")
    assert get_hidden_form_fields(soup) == {}


def test_get_hidden_form_fields_empty_value_included() -> None:
    html = '<form><input type="hidden" name="flag" value=""></form>'
    soup = BeautifulSoup(html, "html.parser")
    fields = get_hidden_form_fields(soup)
    assert fields == {"flag": ""}


def test_get_hidden_form_fields_skips_inputs_without_name() -> None:
    html = '<form><input type="hidden" value="no-name"></form>'
    soup = BeautifulSoup(html, "html.parser")
    assert get_hidden_form_fields(soup) == {}
