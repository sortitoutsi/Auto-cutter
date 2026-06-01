#!/usr/bin/env python3
"""
Web GUI for rating background-removal outputs.

Run:
  ../venv_bg/bin/python rater.py
Then open http://localhost:5050
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

SUBSET_DIR = Path("ab_testing/subset")
AB_DIR = Path("ab_testing")
VARIANTS = [
    "v1_baseline",
    "v2_all_improvements",
    "v2_no_decontam",
    "v2_no_power_curve",
    "v2_aggressive",
    "v3_full",
]
SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
RATINGS_FILE = AB_DIR / "ratings.json"


def load_ratings() -> dict:
    if RATINGS_FILE.exists():
        return json.loads(RATINGS_FILE.read_text())
    return {}


def save_ratings(ratings: dict) -> None:
    RATINGS_FILE.write_text(json.dumps(ratings, indent=2))


def image_list() -> list[str]:
    return sorted(f.name for f in SUBSET_DIR.iterdir() if f.suffix.lower() in SUPPORTED)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("rater.html")


@app.get("/api/images")
def api_images():
    return jsonify(image_list())


@app.get("/api/variants")
def api_variants():
    # Only return variants that have at least one output file
    available = [v for v in VARIANTS if any((AB_DIR / v).glob("*.png"))]
    return jsonify(available)


@app.get("/api/source/<path:name>")
def api_source(name: str):
    path = SUBSET_DIR / name
    if not path.exists():
        return "", 404
    return send_file(path)


@app.get("/api/result/<variant>/<path:name>")
def api_result(variant: str, name: str):
    stem = Path(name).stem
    path = AB_DIR / variant / f"{stem}.png"
    if not path.exists():
        return "", 404
    return send_file(path)


@app.post("/api/rate")
def api_rate():
    data = request.get_json(force=True)
    image = data.get("image", "").strip()
    variant = data.get("variant", "").strip()
    rating = int(data.get("rating", 0))
    if not image or not variant or not (1 <= rating <= 5):
        return jsonify({"error": "invalid"}), 400
    ratings = load_ratings()
    key = f"{image}::{variant}"
    ratings[key] = {
        "image": image,
        "variant": variant,
        "rating": rating,
        "timestamp": datetime.now().isoformat(),
    }
    save_ratings(ratings)
    return jsonify({"ok": True})


@app.get("/api/ratings")
def api_ratings():
    return jsonify(load_ratings())


@app.get("/api/summary")
def api_summary():
    ratings = load_ratings()
    totals: dict[str, list[int]] = {}
    for entry in ratings.values():
        v = entry["variant"]
        totals.setdefault(v, []).append(entry["rating"])
    summary = {
        v: {"count": len(rs), "avg": round(sum(rs) / len(rs), 2)}
        for v, rs in totals.items()
    }
    return jsonify(summary)


if __name__ == "__main__":
    app.run(port=5050, debug=True)
