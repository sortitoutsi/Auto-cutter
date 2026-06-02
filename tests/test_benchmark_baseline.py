"""Tests for `benchmark.load_baseline` schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_cropper.errors import ValidationError
from image_cropper.pipeline.benchmark import load_baseline


def _write(tmp_path: Path, payload: object) -> Path:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(payload))
    return p


VALID_METRICS = {
    "fg_coverage_pct": 50.0,
    "fringe_density_pct": 2.0,
    "mean_fringe_brightness": 100.0,
    "alpha_edge_sharpness": 30.0,
    "h_center_of_mass": 0.5,
    "v_center_of_mass": 0.5,
}


def test_load_baseline_accepts_valid(tmp_path: Path) -> None:
    p = _write(tmp_path, {"img.png": VALID_METRICS})
    out = load_baseline(p)
    assert out == {"img.png": VALID_METRICS}


def test_load_baseline_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        load_baseline(tmp_path / "missing.json")


def test_load_baseline_rejects_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "baseline.json"
    p.write_text("{not json}")
    with pytest.raises(ValidationError, match="not valid JSON"):
        load_baseline(p)


def test_load_baseline_rejects_non_object_root(tmp_path: Path) -> None:
    p = _write(tmp_path, [1, 2, 3])
    with pytest.raises(ValidationError, match="must be a JSON object"):
        load_baseline(p)


def test_load_baseline_rejects_non_object_entry(tmp_path: Path) -> None:
    p = _write(tmp_path, {"img.png": "not a dict"})
    with pytest.raises(ValidationError, match="must be an object"):
        load_baseline(p)


def test_load_baseline_rejects_missing_keys(tmp_path: Path) -> None:
    incomplete = {k: v for k, v in VALID_METRICS.items() if k != "fg_coverage_pct"}
    p = _write(tmp_path, {"img.png": incomplete})
    with pytest.raises(ValidationError, match="missing required keys"):
        load_baseline(p)


def test_load_baseline_rejects_non_numeric_value(tmp_path: Path) -> None:
    broken = {**VALID_METRICS, "fg_coverage_pct": "fifty"}
    p = _write(tmp_path, {"img.png": broken})
    with pytest.raises(ValidationError, match="must be numeric"):
        load_baseline(p)


def test_load_baseline_rejects_bool_value(tmp_path: Path) -> None:
    """bool is a subclass of int — explicitly reject."""
    broken = {**VALID_METRICS, "h_center_of_mass": True}
    p = _write(tmp_path, {"img.png": broken})
    with pytest.raises(ValidationError, match="must be numeric"):
        load_baseline(p)


def test_load_baseline_accepts_committed_baseline() -> None:
    """The repo's actual baseline.json must always pass validation."""
    p = Path(__file__).resolve().parent.parent / "benchmarks" / "baseline.json"
    if not p.exists():
        pytest.skip("committed baseline.json missing")
    out = load_baseline(p)
    assert out  # non-empty
