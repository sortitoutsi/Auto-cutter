"""Tests for `pipeline.benchmark.compute_metrics`.

Uses the committed `benchmarks/golden/*.png` fixtures (already required
by CI) and the committed `benchmarks/baseline.json` to verify the
metric extractor produces the same numbers it always has — this is a
zero-cost regression test on the benchmark machinery itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_cropper.pipeline.benchmark import compute_metrics


@pytest.fixture(scope="session")
def baseline_metrics() -> dict[str, dict[str, float]]:
    p = Path(__file__).resolve().parent.parent / "benchmarks" / "baseline.json"
    if not p.exists():
        pytest.skip(f"baseline.json missing: {p}")
    with p.open() as f:
        return json.load(f)


def test_compute_metrics_returns_all_keys(golden_dark_portrait: Path) -> None:
    m = compute_metrics(golden_dark_portrait)
    expected = {
        "fg_coverage_pct",
        "fringe_density_pct",
        "mean_fringe_brightness",
        "alpha_edge_sharpness",
        "h_center_of_mass",
        "v_center_of_mass",
    }
    assert set(m.keys()) == expected
    for v in m.values():
        assert isinstance(v, float)


def test_compute_metrics_values_in_range(golden_dark_portrait: Path) -> None:
    m = compute_metrics(golden_dark_portrait)
    assert 0.0 <= m["fg_coverage_pct"] <= 100.0
    assert 0.0 <= m["fringe_density_pct"] <= 100.0
    assert 0.0 <= m["mean_fringe_brightness"] <= 255.0
    assert m["alpha_edge_sharpness"] >= 0.0
    assert 0.0 <= m["h_center_of_mass"] <= 1.0
    assert 0.0 <= m["v_center_of_mass"] <= 1.0


def test_compute_metrics_matches_baseline_dark(
    golden_dark_portrait: Path, baseline_metrics: dict[str, dict[str, float]]
) -> None:
    name = golden_dark_portrait.name
    if name not in baseline_metrics:
        pytest.skip(f"{name} not in baseline")
    computed = compute_metrics(golden_dark_portrait)
    expected = baseline_metrics[name]
    for key, want in expected.items():
        got = computed[key]
        # Generous tolerance — different scipy / numpy / PIL minor versions
        # in CI vs dev may drift metrics by tiny amounts.
        assert got == pytest.approx(want, abs=0.5), f"{key}: got {got}, baseline {want}"


def test_compute_metrics_matches_baseline_light(
    golden_light_portrait: Path, baseline_metrics: dict[str, dict[str, float]]
) -> None:
    name = golden_light_portrait.name
    if name not in baseline_metrics:
        pytest.skip(f"{name} not in baseline")
    computed = compute_metrics(golden_light_portrait)
    expected = baseline_metrics[name]
    for key, want in expected.items():
        got = computed[key]
        assert got == pytest.approx(want, abs=0.5), f"{key}: got {got}, baseline {want}"
