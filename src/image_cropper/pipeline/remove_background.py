#!/usr/bin/env python3
"""
High-quality background removal → transparent PNG.

Keeps the entire salient person (face, hair, jersey/kit/shirt). Does not crop.
Optimised for the trickiest part: clean hair edges.

Pipeline (all on GPU when MPS / CUDA is available):
  1. BiRefNet_HR-matting   (2048x2048, soft alpha — gold for hair)
  2. BiRefNet_HR           (2048x2048, salient detection — sharp body)
  3. Horizontal-flip TTA on both
  4. Mean-ensemble the four alpha maps
  5. Bicubic upsample to the original resolution
  6. Kornia fast guided filter (alpha refined by original RGB — locks edges to hair)
  7. Compose RGBA, save lossless PNG

Defaults are tuned for max quality, not speed. Slowness is expected.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from image_cropper.errors import BackgroundRemovalError, ImageCropperError, ValidationError

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

MATTING_MODEL_ID: str = "ZhengPeng7/BiRefNet_HR-matting"
SALIENT_MODEL_ID: str = "ZhengPeng7/BiRefNet_HR"
DEFAULT_INPUT_SIZE: int = 2048

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Device + model loading
# ---------------------------------------------------------------------------


def select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "mps":
        return torch.device("mps")
    if arg == "cuda":
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device) -> Any:
    """Load a BiRefNet model and move it to ``device``.

    Raises :class:`BackgroundRemovalError` if the transformers stack
    cannot be imported or the model cannot be fetched.
    """
    try:
        from transformers import AutoModelForImageSegmentation
    except ImportError as e:
        raise BackgroundRemovalError(
            "transformers is required for background removal; "
            "install with `pip install transformers`"
        ) from e

    print(f"  Loading {model_id} ...", flush=True)
    try:
        model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        raise BackgroundRemovalError(f"failed to load model '{model_id}': {e}") from e
    # Force fp32 — checkpoints sometimes ship with mixed-precision buffers that
    # break MPS inference. Quality > speed here.
    model = model.float().to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image prep
# ---------------------------------------------------------------------------


def pad_to_square(
    image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Letterbox-pad to a square, return (padded, (left, top, right, bottom))."""
    w, h = image.size
    side = max(w, h)
    left = (side - w) // 2
    top = (side - h) // 2
    right = side - w - left
    bottom = side - h - top
    padded = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)
    assert padded.size == (side, side), (
        f"pad_to_square produced {padded.size}, expected square {side}"
    )
    return padded, (left, top, right, bottom)


_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def to_model_tensor(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    out: torch.Tensor = _normalize(t.squeeze(0)).unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_single(
    image_padded: Image.Image, model: Any, device: torch.device, flip: bool, size: int
) -> torch.Tensor:
    """Return one [size, size] alpha tensor on `device`, in [0,1]."""
    src = ImageOps.mirror(image_padded) if flip else image_padded
    x = to_model_tensor(src, size, device)
    out = model(x)
    # BiRefNet returns a list of multi-scale logits; the last is final output.
    logits = out[-1] if isinstance(out, (list, tuple)) else out
    alpha = torch.sigmoid(logits.float())[0, 0]
    if flip:
        alpha = torch.flip(alpha, dims=[-1])
    assert alpha.dim() == 2, f"predict_single returned non-2D tensor: {alpha.shape}"
    return alpha


@torch.no_grad()
def predict_alpha(
    image_rgb: Image.Image,
    models: Sequence[Any] | Iterable[Any],
    device: torch.device,
    tta_flip: bool,
    size: int,
) -> torch.Tensor:
    """Run all models on the (padded) image, optionally with flip TTA. Returns alpha at padded resolution.

    Raises :class:`BackgroundRemovalError` if no models are supplied.
    """
    model_list = list(models)
    if not model_list:
        raise BackgroundRemovalError("predict_alpha requires at least one model")
    padded, _ = pad_to_square(image_rgb)
    acc: torch.Tensor | None = None
    count = 0
    for model in model_list:
        a = predict_single(padded, model, device, flip=False, size=size)
        acc = a if acc is None else acc + a
        count += 1
        if tta_flip:
            a_flip = predict_single(padded, model, device, flip=True, size=size)
            acc = acc + a_flip
            count += 1
    assert acc is not None and count > 0, "no models contributed to alpha — unreachable"
    return acc / count


def crop_padding(alpha: torch.Tensor, image_rgb: Image.Image) -> torch.Tensor:
    """Remove letterbox padding from a square alpha.

    Upsamples alpha to the padded image's full resolution, then crops back
    to the original image size.
    """
    w, h = image_rgb.size
    side = max(w, h)
    alpha_full = F.interpolate(
        alpha[None, None], size=(side, side), mode="bicubic", align_corners=False, antialias=True
    )[0, 0]
    left = (side - w) // 2
    top = (side - h) // 2
    out = alpha_full[top : top + h, left : left + w].clamp(0, 1)
    assert out.shape == (h, w), f"crop_padding produced {tuple(out.shape)}, expected ({h},{w})"
    return out


# ---------------------------------------------------------------------------
# Edge refinement (multi-channel guided filter against the original RGB)
# ---------------------------------------------------------------------------


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable box filter via cumulative sums. x: [..., H, W]. Same shape out."""
    k = 2 * radius + 1
    # avg_pool2d wants 4D. Promote / demote as needed.
    orig_shape = x.shape
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    out = F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius, count_include_pad=False)
    return out.view(orig_shape)


def guided_filter_multichannel(
    guide_chw: torch.Tensor, src_hw: torch.Tensor, radius: int, eps: float
) -> torch.Tensor:
    """3-channel guided filter (He et al. 2010, matrix form).

    guide_chw: [3, H, W] float in [0, 1]
    src_hw:    [H, W]    float in [0, 1]
    returns:   [H, W]    float in [0, 1]
    """
    assert guide_chw.shape[0] == 3
    H, W = src_hw.shape
    img = guide_chw  # [3, H, W] — guide image (variable named per He et al. 2010)
    p = src_hw  # [H, W]

    mean_I = _box_filter(img, radius)  # [3, H, W]
    mean_p = _box_filter(p, radius)  # [H, W]
    mean_Ip = _box_filter(img * p[None], radius)  # [3, H, W]
    cov_Ip = mean_Ip - mean_I * mean_p[None]  # [3, H, W]

    # 3x3 covariance of I (symmetric): rr rg rb gg gb bb
    rr = _box_filter(img[0] * img[0], radius) - mean_I[0] * mean_I[0]
    rg = _box_filter(img[0] * img[1], radius) - mean_I[0] * mean_I[1]
    rb = _box_filter(img[0] * img[2], radius) - mean_I[0] * mean_I[2]
    gg = _box_filter(img[1] * img[1], radius) - mean_I[1] * mean_I[1]
    gb = _box_filter(img[1] * img[2], radius) - mean_I[1] * mean_I[2]
    bb = _box_filter(img[2] * img[2], radius) - mean_I[2] * mean_I[2]

    # Build per-pixel 3x3 covariance + eps*I and invert analytically.
    rr_e = rr + eps
    gg_e = gg + eps
    bb_e = bb + eps

    # Cofactors / determinant of symmetric 3x3
    inv_rr = gg_e * bb_e - gb * gb
    inv_rg = gb * rb - rg * bb_e
    inv_rb = rg * gb - gg_e * rb
    inv_gg = rr_e * bb_e - rb * rb
    inv_gb = rg * rb - rr_e * gb
    inv_bb = rr_e * gg_e - rg * rg

    det = rr_e * inv_rr + rg * inv_rg + rb * inv_rb
    det = det.clamp_min(1e-12)
    inv_rr = inv_rr / det
    inv_rg = inv_rg / det
    inv_rb = inv_rb / det
    inv_gg = inv_gg / det
    inv_gb = inv_gb / det
    inv_bb = inv_bb / det

    # a = Σ^-1 · cov_Ip  (per-pixel matrix-vector product)
    c0, c1, c2 = cov_Ip[0], cov_Ip[1], cov_Ip[2]
    a0 = inv_rr * c0 + inv_rg * c1 + inv_rb * c2
    a1 = inv_rg * c0 + inv_gg * c1 + inv_gb * c2
    a2 = inv_rb * c0 + inv_gb * c1 + inv_bb * c2

    b = mean_p - (a0 * mean_I[0] + a1 * mean_I[1] + a2 * mean_I[2])

    mean_a0 = _box_filter(a0, radius)
    mean_a1 = _box_filter(a1, radius)
    mean_a2 = _box_filter(a2, radius)
    mean_b = _box_filter(b, radius)

    q = mean_a0 * img[0] + mean_a1 * img[1] + mean_a2 * img[2] + mean_b
    return q.clamp(0, 1)


def guided_refine(
    alpha_hw: torch.Tensor, image_rgb: Image.Image, device: torch.device, radius: int, eps: float
) -> torch.Tensor:
    """Refine alpha so its edges follow the image's edges (hair, fly-aways).

    Runs on CPU — Kornia's MPS path triggers an IOGPU shmem assert on macOS, and
    the filter is fast enough on CPU (~few hundred ms for a 1000²-ish image).
    """
    img_np = np.asarray(image_rgb.convert("RGB"), dtype=np.float32) / 255.0
    guide = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # [3, H, W] CPU
    src = alpha_hw.detach().to("cpu").float()  # [H, W] CPU
    return guided_filter_multichannel(guide, src, radius=radius, eps=eps)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def apply_alpha(image_rgb: Image.Image, alpha_hw: torch.Tensor) -> Image.Image:
    w, h = image_rgb.size
    assert alpha_hw.shape == (h, w), f"alpha {tuple(alpha_hw.shape)} != image (H,W)=({h},{w})"
    a = (alpha_hw.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    rgba = image_rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(a, mode="L"))
    return rgba


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_image(
    input_path: Path,
    output_path: Path,
    models: Sequence[Any] | Iterable[Any],
    device: torch.device,
    tta_flip: bool,
    refine: bool,
    refine_radius: int,
    refine_eps: float,
    input_size: int,
) -> None:
    """Remove the background from one image. Raises BackgroundRemovalError on failure."""
    image = Image.open(input_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    print(f"  Image: {w}x{h}")

    t0 = time.perf_counter()
    alpha_lowres = predict_alpha(image, models, device, tta_flip=tta_flip, size=input_size)
    alpha = crop_padding(alpha_lowres, image)
    if refine:
        alpha = guided_refine(alpha, image, device, radius=refine_radius, eps=refine_eps)
    rgba = apply_alpha(image, alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path, format="PNG", optimize=True, compress_level=9)
    dt = time.perf_counter() - t0
    print(f"  Saved → {output_path}  ({dt:.1f}s)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default="cropped", help="Input directory (default: cropped)")
    p.add_argument(
        "--output",
        default="output/transparent",
        help="Output directory (default: output/transparent)",
    )
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument(
        "--no-tta", action="store_true", help="Disable horizontal-flip test-time augmentation"
    )
    p.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Use only BiRefNet_HR-matting (skip salient model)",
    )
    p.add_argument("--no-refine", action="store_true", help="Disable guided-filter edge refinement")
    p.add_argument(
        "--refine-radius", type=int, default=4, help="Guided-filter radius in pixels (default: 4)"
    )
    p.add_argument(
        "--refine-eps",
        type=float,
        default=1e-4,
        help="Guided-filter regularisation eps (default: 1e-4)",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help=f"Model input size (default: {DEFAULT_INPUT_SIZE}). Lower if MPS runs out of memory.",
    )
    p.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all)")
    p.add_argument(
        "--overwrite", action="store_true", help="Re-process even if output already exists"
    )
    args = p.parse_args()

    try:
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        if not input_dir.exists():
            raise ValidationError(f"input directory not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        device = select_device(args.device)
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

        images = [
            p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if args.limit > 0:
            images = images[: args.limit]
        if not images:
            print(f"No supported images in {input_dir}/")
            sys.exit(0)

        print(f"Device: {device}")
        print(f"Found {len(images)} image(s) in {input_dir}/")
        print(
            f"Settings: tta_flip={not args.no_tta}, ensemble={not args.no_ensemble}, "
            f"refine={not args.no_refine} (radius={args.refine_radius}, eps={args.refine_eps})\n"
        )

        models = [load_model(MATTING_MODEL_ID, device)]
        if not args.no_ensemble:
            models.append(load_model(SALIENT_MODEL_ID, device))
        print()

        ok = 0
        skipped = 0
        for i, img_path in enumerate(images, 1):
            out_path = output_dir / (img_path.stem + ".png")
            if out_path.exists() and not args.overwrite:
                print(f"[{i}/{len(images)}] {img_path.name} — exists, skipping (use --overwrite)")
                skipped += 1
                continue
            print(f"[{i}/{len(images)}] {img_path.name}")
            try:
                process_image(
                    img_path,
                    out_path,
                    models,
                    device,
                    tta_flip=not args.no_tta,
                    refine=not args.no_refine,
                    refine_radius=args.refine_radius,
                    refine_eps=args.refine_eps,
                    input_size=args.input_size,
                )
                ok += 1
            except ImageCropperError as e:
                print(f"  [!] {e}")
            except Exception as e:
                print(f"  [!] unexpected error: {e}", file=sys.stderr)
            print()

        print(f"Done. {ok} processed, {skipped} skipped, {len(images) - ok - skipped} failed.")
    except ImageCropperError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
