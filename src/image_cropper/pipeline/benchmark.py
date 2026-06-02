#!/usr/bin/env python3
"""
Portrait pipeline quality benchmark.

Computes alpha-channel quality metrics on PNG outputs — no ML models required.
Metrics are stable, deterministic, and comparable across model/algorithm versions.

Usage:
  # compare a directory against committed baseline
  python scripts/benchmark.py benchmarks/golden/ --compare benchmarks/baseline.json

  # just print metrics for a set of outputs
  python scripts/benchmark.py output/final/

  # promote current results as the new baseline
  python scripts/benchmark.py output/final/ --update-baseline benchmarks/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED = {".png", ".webp"}
DEFAULT_BASELINE = Path(__file__).parent.parent / "benchmarks" / "baseline.json"

# Alpha thresholds — must stay in sync with deglow.py constants
_ALPHA_FG: int = 127  # pixel counts as foreground
_FRINGE_MIN: int = 5  # ignore near-invisible dust below this
_FRINGE_MAX: int = 219  # pixel is fringe if alpha < this (mirrors deglow OPAQUE_MIN-1)

# Maximum allowed absolute delta from baseline before the check fails.
# Tuned so small algorithm tweaks pass but real regressions do not.
TOLERANCES: dict[str, float] = {
    "fg_coverage_pct": 2.0,  # ± 2 pp foreground area
    "fringe_density_pct": 1.5,  # ± 1.5 pp fringe pixels
    "mean_fringe_brightness": 10.0,  # ± 10/255 luminance
    "alpha_edge_sharpness": 8.0,  # ± 8 RMS gradient units
    "h_center_of_mass": 0.05,  # ± 5 % of canvas width off-center
    "v_center_of_mass": 0.05,  # ± 5 % of canvas height off-center
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luminance, input shape (..., 3) in 0-255, output in 0-255."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def compute_metrics(img_path: Path) -> dict[str, float]:
    """Return a flat dict of quality metrics for a single PNG/WebP image."""
    img = Image.open(img_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img, dtype=np.float32)
    rgb = arr[:, :, :3]  # (H, W, 3)
    alpha = arr[:, :, 3]  # (H, W)

    H, W = alpha.shape
    total = H * W

    # Foreground coverage: fraction of pixels the model kept as subject
    fg_mask = alpha > _ALPHA_FG
    fg_coverage_pct = float(fg_mask.sum()) / total * 100.0

    # Fringe density: semi-transparent pixels (halo / edge bleed indicator)
    fringe_mask = (alpha >= _FRINGE_MIN) & (alpha <= _FRINGE_MAX)
    fringe_density_pct = float(fringe_mask.sum()) / total * 100.0

    # Mean brightness of fringe pixels — high = glow still present
    if fringe_mask.any():
        mean_fringe_brightness = float(_luminance(rgb[fringe_mask]).mean())
    else:
        mean_fringe_brightness = 0.0

    # Alpha edge sharpness: RMS of the alpha gradient at strong edges.
    # Higher = crisper transitions (better background removal).
    gx = np.abs(np.diff(alpha, axis=1))  # (H, W-1)
    gy = np.abs(np.diff(alpha, axis=0))  # (H-1, W)
    threshold = 10.0
    strong = np.concatenate([gx[gx > threshold].ravel(), gy[gy > threshold].ravel()])
    alpha_edge_sharpness = float(np.sqrt(np.mean(strong**2))) if len(strong) else 0.0

    # Center of mass of the alpha channel — measures subject centering
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    alpha_sum = alpha.sum()
    if alpha_sum > 0:
        h_center_of_mass = float((xs * alpha).sum() / alpha_sum / W)
        v_center_of_mass = float((ys * alpha).sum() / alpha_sum / H)
    else:
        h_center_of_mass = 0.5
        v_center_of_mass = 0.5

    return {
        "fg_coverage_pct": round(fg_coverage_pct, 3),
        "fringe_density_pct": round(fringe_density_pct, 3),
        "mean_fringe_brightness": round(mean_fringe_brightness, 3),
        "alpha_edge_sharpness": round(alpha_edge_sharpness, 3),
        "h_center_of_mass": round(h_center_of_mass, 4),
        "v_center_of_mass": round(v_center_of_mass, 4),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _delta_str(value: float, base: float, tolerance: float) -> str:
    delta = value - base
    sign = "+" if delta >= 0 else ""
    flag = " FAIL" if abs(delta) > tolerance else "     "
    return f"{flag}  (baseline {base:.3f}, Δ {sign}{delta:.3f}, tol ±{tolerance})"


def print_results(results: dict[str, dict], baseline: dict[str, dict] | None) -> None:
    for img_name, metrics in sorted(results.items()):
        print(f"\n  {img_name}")
        base = (baseline or {}).get(img_name, {})
        for k, v in metrics.items():
            annotation = ""
            if k in base:
                annotation = _delta_str(v, base[k], TOLERANCES.get(k, 1.0))
            print(f"    {k:<30} {v:>10.3f}{annotation}")


def compare_to_baseline(results: dict[str, dict], baseline: dict[str, dict]) -> list[str]:
    failures = []
    for img_name, metrics in results.items():
        if img_name not in baseline:
            continue
        base = baseline[img_name]
        for metric, value in metrics.items():
            if metric not in base:
                continue
            tolerance = TOLERANCES.get(metric, 1.0)
            if abs(value - base[metric]) > tolerance:
                delta = value - base[metric]
                sign = "+" if delta >= 0 else ""
                failures.append(
                    f"{img_name}/{metric}: got {value:.3f}, "
                    f"baseline {base[metric]:.3f}, "
                    f"Δ {sign}{delta:.3f} (tol ±{tolerance})"
                )
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute image quality metrics on pipeline PNG outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image_dir", type=Path, help="Directory of PNG/WebP images to benchmark")
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="BASELINE_JSON",
        help="Compare results against this baseline and fail on regression",
    )
    parser.add_argument(
        "--update-baseline",
        type=Path,
        metavar="BASELINE_JSON",
        help="Write computed metrics as the new baseline to this file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="RESULTS_JSON",
        help="Also write raw results JSON to this file",
    )
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        print(f"error: '{args.image_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not images:
        print(f"No PNG/WebP images found in {args.image_dir}")
        sys.exit(1)

    print(f"Benchmarking {len(images)} image(s) in {args.image_dir}")
    results: dict[str, dict] = {}
    for img_path in images:
        print(f"  {img_path.name} ...", end=" ", flush=True)
        results[img_path.name] = compute_metrics(img_path)
        print("ok")

    baseline: dict[str, dict] | None = None
    if args.compare:
        if not args.compare.exists():
            print(f"warning: baseline file '{args.compare}' not found — skipping comparison")
        else:
            with open(args.compare) as f:
                baseline = json.load(f)

    print_results(results, baseline)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    if args.update_baseline:
        args.update_baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.update_baseline, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline updated: {args.update_baseline}")

    if baseline:
        failures = compare_to_baseline(results, baseline)
        if failures:
            print(f"\n{len(failures)} REGRESSION(S) DETECTED:")
            for fail in failures:
                print(f"  FAIL  {fail}")
            sys.exit(1)
        total_checks = sum(len(m) for m in results.values())
        print(f"\nAll {total_checks} metric checks passed.")


if __name__ == "__main__":
    main()
