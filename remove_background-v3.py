#!/usr/bin/env python3
"""
High-quality background removal → transparent PNG. (v3)

Designed for ~29-minute-per-image compute budgets (e.g. 50 images/day batch).

Pipeline:
  1. Multi-model ensemble  — configurable list of BiRefNet variants
  2. Multi-scale TTA       — each model run at multiple input sizes + H-flip
  3. Mean ensemble alpha   — average of all passes
  4. Trimap generation     — hard fg/bg thresholds + dilation to create unknown band
  5. PyMatting refinement  — closed-form alpha matting in the unknown zone
  6. Iterative guided filter — N passes with tightening eps (sharper each pass)
  7. Alpha power-curve     — tighten semi-transparent fringe toward 0
  8. Decontamination       — remove background colour spill from edge pixels
  9. Compose RGBA + save lossless PNG

v3 vs v2:
  - 3-model default ensemble  (added ZhengPeng7/BiRefNet for 1024-native diversity)
  - multi-scale TTA at [1024, 2048] — 12 inference passes by default vs 4
  - trimap + PyMatting closed-form matting for sub-pixel hair/edge accuracy
  - iterative guided filter (3 passes, eps tightened each pass)
  - per-step timing printed so you can see where budget is spent
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Default ensemble — ordered by quality contribution.
# BiRefNet_HR-matting: soft alpha, gold for hair
# BiRefNet_HR:         salient detection, sharp body outlines
# BiRefNet:            1024-native general model, adds scale diversity
DEFAULT_MODEL_IDS = [
    "ZhengPeng7/BiRefNet_HR-matting",
    "ZhengPeng7/BiRefNet_HR",
    "ZhengPeng7/BiRefNet",
]

# Input sizes at which every model is run (multi-scale TTA).
DEFAULT_SCALES = [1024, 2048]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(label: str, t0: float) -> float:
    dt = time.perf_counter() - t0
    print(f"    [{label}] {dt:.1f}s", flush=True)
    return time.perf_counter()


def select_device(arg: str) -> torch.device:
    if arg == "cpu":   return torch.device("cpu")
    if arg == "mps":   return torch.device("mps")
    if arg == "cuda":  return torch.device("cuda")
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device):
    from transformers import AutoModelForImageSegmentation
    print(f"    {model_id}", flush=True)
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model = model.float().to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image prep
# ---------------------------------------------------------------------------

def pad_to_square(image: Image.Image, fill=(255, 255, 255)) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = image.size
    side = max(w, h)
    left   = (side - w) // 2
    top    = (side - h) // 2
    right  = side - w - left
    bottom = side - h - top
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill), (left, top, right, bottom)


def to_model_tensor(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    return _normalize(t.squeeze(0)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Inference — multi-model, multi-scale TTA
# ---------------------------------------------------------------------------

@torch.no_grad()
def _single_pass(padded: Image.Image, model, device: torch.device, size: int, flip: bool) -> torch.Tensor:
    src = ImageOps.mirror(padded) if flip else padded
    x   = to_model_tensor(src, size, device)
    out = model(x)
    logits = out[-1] if isinstance(out, (list, tuple)) else out
    alpha  = torch.sigmoid(logits.float())[0, 0]
    if flip:
        alpha = torch.flip(alpha, dims=[-1])
    return alpha  # [size, size]


@torch.no_grad()
def predict_alpha_ensemble(
    image_rgb: Image.Image,
    models: list,
    device: torch.device,
    scales: list[int],
    tta_flip: bool,
) -> torch.Tensor:
    """Run every model at every scale (+flip TTA). Return mean alpha at max scale."""
    padded, _ = pad_to_square(image_rgb)  # white fill
    max_size   = max(scales)
    acc        = None
    count      = 0
    for model in models:
        for size in scales:
            a = _single_pass(padded, model, device, size, flip=False)
            # Upsample smaller-scale predictions to max_size before averaging
            if a.shape[-1] != max_size:
                a = F.interpolate(a[None, None], size=(max_size, max_size),
                                  mode="bicubic", align_corners=False, antialias=True)[0, 0]
            acc   = a if acc is None else acc + a
            count += 1
            if tta_flip:
                af = _single_pass(padded, model, device, size, flip=True)
                if af.shape[-1] != max_size:
                    af = F.interpolate(af[None, None], size=(max_size, max_size),
                                       mode="bicubic", align_corners=False, antialias=True)[0, 0]
                acc    = acc + af
                count += 1
    return acc / count  # [max_size, max_size]


def crop_padding(alpha: torch.Tensor, image_rgb: Image.Image) -> torch.Tensor:
    w, h   = image_rgb.size
    side   = max(w, h)
    a_size = alpha.shape[-1]
    alpha_full = F.interpolate(
        alpha[None, None], size=(side, side), mode="bicubic", align_corners=False, antialias=True
    )[0, 0]
    left = (side - w) // 2
    top  = (side - h) // 2
    return alpha_full[top:top + h, left:left + w].clamp(0, 1)


# ---------------------------------------------------------------------------
# Trimap generation
# ---------------------------------------------------------------------------

def make_trimap(
    alpha_np: np.ndarray,
    fg_thresh: float = 0.85,
    bg_thresh: float = 0.15,
    dilation_px: int = 15,
) -> np.ndarray:
    """Create a trimap: 1.0 = definite fg, 0.0 = definite bg, 0.5 = unknown.

    The unknown band is formed by dilating the fg/bg boundary so that PyMatting
    has enough context to recover sub-pixel detail.
    """
    from scipy.ndimage import binary_dilation

    fg_hard = alpha_np > fg_thresh
    bg_hard = alpha_np < bg_thresh

    struct   = np.ones((3, 3), dtype=bool)
    fg_dilated = binary_dilation(fg_hard, structure=struct, iterations=dilation_px)
    bg_dilated = binary_dilation(bg_hard, structure=struct, iterations=dilation_px)

    unknown = fg_dilated & bg_dilated

    trimap            = np.zeros_like(alpha_np, dtype=np.float64)
    trimap[fg_hard]   = 1.0
    trimap[unknown]   = 0.5
    # bg stays 0; fg overrides unknown where alpha was hard fg
    trimap[fg_hard & unknown] = 1.0
    return trimap


# ---------------------------------------------------------------------------
# PyMatting refinement
# ---------------------------------------------------------------------------

def pymatting_refine(
    image_rgb: Image.Image,
    alpha_hw: torch.Tensor,
    fg_thresh: float,
    bg_thresh: float,
    dilation_px: int,
    max_size: int,
    method: str,
) -> torch.Tensor:
    """Closed-form (or random-walk) alpha matting in the trimap unknown zone.

    Downsamples to max_size for speed, then upsampls result back to original.
    Falls back to the input alpha on any error.
    """
    try:
        from pymatting import estimate_alpha_cf, estimate_alpha_rw
    except ImportError:
        print("    [matting] pymatting not installed — skipping", flush=True)
        return alpha_hw

    try:
        w, h = image_rgb.size

        # Optionally downsample for speed
        scale  = min(1.0, max_size / max(w, h))
        mw, mh = int(w * scale), int(h * scale)
        img_small = image_rgb.resize((mw, mh), Image.LANCZOS)

        # Alpha at matting resolution
        a_small = F.interpolate(
            alpha_hw[None, None].float(), size=(mh, mw),
            mode="bicubic", align_corners=False, antialias=True
        )[0, 0].cpu().numpy()

        img_np  = np.asarray(img_small.convert("RGB"), dtype=np.float64) / 255.0
        trimap  = make_trimap(a_small, fg_thresh, bg_thresh, max(1, int(dilation_px * scale)))

        import warnings
        estimator = estimate_alpha_rw if method == "rw" else estimate_alpha_cf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha_mat = estimator(img_np, trimap).astype(np.float32)

        # Upsample back to original size if we downsampled
        if scale < 1.0:
            alpha_mat = np.array(
                Image.fromarray(alpha_mat).resize((w, h), Image.LANCZOS)
            )

        result = torch.from_numpy(alpha_mat).clamp(0, 1)
        return result

    except Exception as e:
        print(f"    [matting] failed ({e}), keeping neural alpha", flush=True)
        return alpha_hw


# ---------------------------------------------------------------------------
# Edge refinement — guided filter (multi-channel, pure PyTorch)
# ---------------------------------------------------------------------------

def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    orig = x.shape
    if x.dim() == 2: x = x[None, None]
    elif x.dim() == 3: x = x[None]
    out = F.avg_pool2d(x, 2 * radius + 1, stride=1, padding=radius, count_include_pad=False)
    return out.view(orig)


def _guided_filter_once(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """Single-pass 3-channel guided filter on CPU tensors."""
    I, p = guide, src
    mean_I  = _box_filter(I, radius)
    mean_p  = _box_filter(p, radius)
    mean_Ip = _box_filter(I * p[None], radius)
    cov_Ip  = mean_Ip - mean_I * mean_p[None]

    rr = _box_filter(I[0]*I[0], radius) - mean_I[0]*mean_I[0]
    rg = _box_filter(I[0]*I[1], radius) - mean_I[0]*mean_I[1]
    rb = _box_filter(I[0]*I[2], radius) - mean_I[0]*mean_I[2]
    gg = _box_filter(I[1]*I[1], radius) - mean_I[1]*mean_I[1]
    gb = _box_filter(I[1]*I[2], radius) - mean_I[1]*mean_I[2]
    bb = _box_filter(I[2]*I[2], radius) - mean_I[2]*mean_I[2]

    rr_e, gg_e, bb_e = rr + eps, gg + eps, bb + eps
    inv_rr = gg_e*bb_e - gb*gb
    inv_rg = gb*rb   - rg*bb_e
    inv_rb = rg*gb   - gg_e*rb
    inv_gg = rr_e*bb_e - rb*rb
    inv_gb = rg*rb   - rr_e*gb
    inv_bb = rr_e*gg_e - rg*rg

    det = (rr_e*inv_rr + rg*inv_rg + rb*inv_rb).clamp_min(1e-12)
    inv_rr /= det; inv_rg /= det; inv_rb /= det
    inv_gg /= det; inv_gb /= det; inv_bb /= det

    c0, c1, c2 = cov_Ip[0], cov_Ip[1], cov_Ip[2]
    a0 = inv_rr*c0 + inv_rg*c1 + inv_rb*c2
    a1 = inv_rg*c0 + inv_gg*c1 + inv_gb*c2
    a2 = inv_rb*c0 + inv_gb*c1 + inv_bb*c2
    b  = mean_p - (a0*mean_I[0] + a1*mean_I[1] + a2*mean_I[2])

    q = _box_filter(a0,radius)*I[0] + _box_filter(a1,radius)*I[1] + \
        _box_filter(a2,radius)*I[2] + _box_filter(b, radius)
    return q.clamp(0, 1)


def iterative_guided_refine(
    alpha_hw: torch.Tensor,
    image_rgb: Image.Image,
    n_passes: int,
    radius: int,
    eps_start: float,
    eps_end: float,
) -> torch.Tensor:
    """Run the guided filter N times, linearly interpolating eps from coarse→fine."""
    img_np = np.asarray(image_rgb.convert("RGB"), dtype=np.float32) / 255.0
    guide  = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    alpha  = alpha_hw.detach().cpu().float()

    eps_vals = np.linspace(eps_start, eps_end, max(n_passes, 1)).tolist()
    for i, eps in enumerate(eps_vals):
        alpha = _guided_filter_once(guide, alpha, radius=radius, eps=eps)

    return alpha


# ---------------------------------------------------------------------------
# Post-processing (same as v2)
# ---------------------------------------------------------------------------

def alpha_power_curve(alpha: torch.Tensor, gamma: float) -> torch.Tensor:
    return alpha.pow(gamma).clamp(0, 1)


def decontaminate(image_rgb: Image.Image, alpha_hw: torch.Tensor,
                  bg_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    img_np = np.asarray(image_rgb.convert("RGB"), dtype=np.float32)
    a3     = alpha_hw.detach().cpu().numpy().astype(np.float32)[..., None]
    bg     = np.array(bg_color, dtype=np.float32)
    fg     = (img_np - (1.0 - a3) * bg) / np.clip(a3, 1e-6, 1.0)
    return Image.fromarray(np.clip(fg, 0, 255).astype(np.uint8), mode="RGB")


def apply_alpha(image_rgb: Image.Image, alpha_hw: torch.Tensor) -> Image.Image:
    w, h = image_rgb.size
    assert alpha_hw.shape == (h, w)
    a    = (alpha_hw.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    rgba = image_rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(a, mode="L"))
    return rgba


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_image(input_path: Path, output_path: Path, models: list, device,
                  scales: list[int], tta_flip: bool,
                  matting: bool, matting_method: str, matting_max_size: int,
                  matting_fg_thresh: float, matting_bg_thresh: float, matting_dilation: int,
                  refine_passes: int, refine_radius: int, refine_eps_start: float, refine_eps_end: float,
                  alpha_gamma: float, decontam: bool, bg_color: tuple[int, int, int]):

    image = Image.open(input_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    print(f"  Image: {w}x{h} | {len(models)} models × {len(scales)} scales"
          f" × {'2' if tta_flip else '1'} flips = "
          f"{len(models)*len(scales)*(2 if tta_flip else 1)} passes")

    t0 = t = time.perf_counter()

    # 1. Multi-model multi-scale ensemble
    alpha_lowres = predict_alpha_ensemble(image, models, device, scales, tta_flip)
    alpha        = crop_padding(alpha_lowres, image)
    t = _t("ensemble inference", t)

    # 2. Trimap + PyMatting
    if matting:
        alpha = pymatting_refine(image, alpha,
                                 fg_thresh=matting_fg_thresh, bg_thresh=matting_bg_thresh,
                                 dilation_px=matting_dilation, max_size=matting_max_size,
                                 method=matting_method)
        t = _t("pymatting", t)

    # 3. Iterative guided filter
    if refine_passes > 0:
        alpha = iterative_guided_refine(alpha, image,
                                        n_passes=refine_passes, radius=refine_radius,
                                        eps_start=refine_eps_start, eps_end=refine_eps_end)
        t = _t(f"guided filter ×{refine_passes}", t)

    # 4. Power curve
    if alpha_gamma != 1.0:
        alpha = alpha_power_curve(alpha, alpha_gamma)

    # 5. Decontaminate
    if decontam:
        image = decontaminate(image, alpha, bg_color=bg_color)

    rgba = apply_alpha(image, alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path, format="PNG", optimize=True, compress_level=9)
    print(f"  Total: {time.perf_counter()-t0:.1f}s  →  {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",   default="cropped",             help="Input directory")
    p.add_argument("--output",  default="output_transparent",  help="Output directory")
    p.add_argument("--device",  default="auto", choices=["auto","mps","cuda","cpu"])

    # Model ensemble
    p.add_argument("--models", nargs="+", default=DEFAULT_MODEL_IDS,
                   help="HuggingFace model IDs to ensemble (space-separated)")

    # Multi-scale TTA
    p.add_argument("--scales", nargs="+", type=int, default=DEFAULT_SCALES,
                   help="Input sizes to run inference at (default: 1024 2048)")
    p.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip TTA")

    # PyMatting
    p.add_argument("--no-matting",          action="store_true")
    p.add_argument("--matting-method",      default="cf", choices=["cf","rw"],
                   help="cf = closed-form (default, more accurate); rw = random walk (faster)")
    p.add_argument("--matting-max-size",    type=int, default=2000,
                   help="Downsample to this size for matting (speed vs quality, default: 2000)")
    p.add_argument("--matting-fg-thresh",   type=float, default=0.85)
    p.add_argument("--matting-bg-thresh",   type=float, default=0.15)
    p.add_argument("--matting-dilation",    type=int,   default=15,
                   help="Dilation radius (px) for unknown band (default: 15)")

    # Iterative guided filter
    p.add_argument("--refine-passes",     type=int,   default=3,
                   help="Number of guided filter passes (default: 3, 0 = disable)")
    p.add_argument("--refine-radius",     type=int,   default=12)
    p.add_argument("--refine-eps-start",  type=float, default=1e-5,
                   help="Guided filter eps for pass 1 (coarse, default: 1e-5)")
    p.add_argument("--refine-eps-end",    type=float, default=1e-7,
                   help="Guided filter eps for final pass (fine, default: 1e-7)")

    # Post-processing (same as v2)
    p.add_argument("--alpha-gamma",  type=float, default=1.3)
    p.add_argument("--no-decontam",  action="store_true")
    p.add_argument("--bg-color",     type=str, default="255,255,255")

    p.add_argument("--input-size",   type=int,  default=2048, help="(ignored — use --scales)")
    p.add_argument("--limit",        type=int,  default=0)
    p.add_argument("--overwrite",    action="store_true")
    args = p.parse_args()

    try:
        bg_color = tuple(int(x) for x in args.bg_color.split(","))
        assert len(bg_color) == 3
    except Exception:
        print("Error: --bg-color must be R,G,B e.g. 255,255,255", file=sys.stderr)
        sys.exit(1)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        print(f"Error: '{input_dir}' not found", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    images = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        print(f"No supported images in {input_dir}/")
        sys.exit(0)

    n_passes = len(args.models) * len(args.scales) * (2 if not args.no_tta else 1)
    print(f"Device:  {device}")
    print(f"Models:  {args.models}")
    print(f"Scales:  {args.scales}  |  TTA flip: {not args.no_tta}  →  {n_passes} passes/image")
    print(f"Matting: {not args.no_matting} ({args.matting_method}, dilation={args.matting_dilation})")
    print(f"Refine:  {args.refine_passes} passes  eps {args.refine_eps_start}→{args.refine_eps_end}")
    print(f"Post:    gamma={args.alpha_gamma}  decontam={not args.no_decontam}  bg={bg_color}")
    print(f"\nLoading {len(args.models)} model(s)...")

    models = [load_model(mid, device) for mid in args.models]
    print()

    ok = skipped = 0
    for i, img_path in enumerate(images, 1):
        out_path = output_dir / (img_path.stem + ".png")
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(images)}] {img_path.name} — skipping (use --overwrite)")
            skipped += 1
            continue
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            process_image(
                img_path, out_path, models, device,
                scales=args.scales, tta_flip=not args.no_tta,
                matting=not args.no_matting, matting_method=args.matting_method,
                matting_max_size=args.matting_max_size,
                matting_fg_thresh=args.matting_fg_thresh, matting_bg_thresh=args.matting_bg_thresh,
                matting_dilation=args.matting_dilation,
                refine_passes=args.refine_passes, refine_radius=args.refine_radius,
                refine_eps_start=args.refine_eps_start, refine_eps_end=args.refine_eps_end,
                alpha_gamma=args.alpha_gamma, decontam=not args.no_decontam, bg_color=bg_color,
            )
            ok += 1
        except Exception as e:
            import traceback
            print(f"  [!] Error: {e}")
            traceback.print_exc()
        print()

    print(f"Done. {ok} processed, {skipped} skipped, {len(images)-ok-skipped} failed.")


if __name__ == "__main__":
    main()
