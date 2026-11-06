#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PRED_DIR = Path("./Final_Model/DDN-Data/result_DDN")
DEFAULT_GT_DIR = Path("Datasets/DDN-Data/test/target")


@dataclass(frozen=True)
class EvalConfig:
    pred_dir: Path
    gt_dir: Path
    recursive: bool
    quiet: bool
    csv_path: Path
    limit: int | None


@dataclass(frozen=True)
class MetricResult:
    psnr: float
    ssim: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Y-channel PSNR and SSIM for paired images in two directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval_DID_DDN.py\n"
            "  python eval_DID_DDN.py --pred-dir results --gt-dir gt\n"
            "  python eval_DID_DDN.py --csv-path metrics.csv"
        ),
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Directory containing predicted/restored images.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help="Directory containing ground-truth images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively under both directories.",
    )
    default_csv_path = Path(__file__).resolve().parent / "results.csv"
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv_path,
        help="CSV path used to save per-image PSNR and SSIM values.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N matched image pairs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary statistics.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        recursive=args.recursive,
        quiet=args.quiet,
        csv_path=args.csv_path,
        limit=args.limit,
    )


def numeric_sort_key(name: str) -> tuple[int, object]:
    return (0, int(name)) if name.isdigit() else (1, name)


def iter_image_paths(image_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        return sorted((p for p in image_dir.rglob("*") if p.is_file()), key=lambda p: str(p))
    return sorted((p for p in image_dir.iterdir() if p.is_file()), key=lambda p: str(p))


def collect_images(image_dir: Path, recursive: bool) -> dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {image_dir}")

    images: dict[str, Path] = {}
    for path in iter_image_paths(image_dir, recursive):
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.stem in images:
            raise ValueError(
                f"Duplicate image stem detected in {image_dir}: {path.stem!r}. "
                "Use unique file names before evaluation."
            )
        images[path.stem] = path
    return images


def load_image(path: Path, mode: str = "RGB") -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert(mode), dtype=np.float32)
    return arr


def ensure_same_shape(pred: np.ndarray, gt: np.ndarray, pred_path: Path, gt_path: Path) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            "Image size mismatch: "
            f"{pred_path.name} -> {pred.shape}, {gt_path.name} -> {gt.shape}"
        )


def compute_psnr(image_a: np.ndarray, image_b: np.ndarray, data_range: float) -> float:
    mse = np.mean((image_a.astype(np.float64) - image_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


def rgb_to_matlab_y_channel(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return 16.0 + (
        65.481 * rgb[:, :, 0] + 128.553 * rgb[:, :, 1] + 24.966 * rgb[:, :, 2]
    ) / 255.0


def create_gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    radius = size // 2
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    window = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return window / window.sum()


def convolve2d_same_symm(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import convolve2d
    except ImportError as exc:
        raise SystemExit(
            "Mode `y-channel` requires `scipy`. Install it with: pip install scipy"
        ) from exc
    return convolve2d(image, kernel, mode="same", boundary="symm")


def convolve2d_valid_fill(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import convolve2d
    except ImportError as exc:
        raise SystemExit(
            "Mode `y-channel` requires `scipy`. Install it with: pip install scipy"
        ) from exc
    return convolve2d(image, kernel, mode="valid")


def matlab_style_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    window = create_gaussian_window(size=11, sigma=1.5)
    height, width = gray_a.shape
    factor = max(1, round(min(height, width) / 256))

    if factor > 1:
        low_pass = np.ones((factor, factor), dtype=np.float64)
        low_pass /= low_pass.sum()
        gray_a = convolve2d_same_symm(gray_a, low_pass)[::factor, ::factor]
        gray_b = convolve2d_same_symm(gray_b, low_pass)[::factor, ::factor]

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = convolve2d_valid_fill(gray_a, window)
    mu_b = convolve2d_valid_fill(gray_b, window)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = convolve2d_valid_fill(gray_a * gray_a, window) - mu_a_sq
    sigma_b_sq = convolve2d_valid_fill(gray_b * gray_b, window) - mu_b_sq
    sigma_ab = convolve2d_valid_fill(gray_a * gray_b, window) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    )
    return float(ssim_map.mean())


def compute_metrics_y_channel(pred_path: Path, gt_path: Path) -> MetricResult:
    pred = load_image(pred_path)
    gt = load_image(gt_path)
    ensure_same_shape(pred, gt, pred_path, gt_path)
    pred_y = rgb_to_matlab_y_channel(pred)
    gt_y = rgb_to_matlab_y_channel(gt)
    return MetricResult(
        psnr=compute_psnr(pred_y, gt_y, 255.0),
        ssim=matlab_style_ssim(pred_y, gt_y),
    )


def save_csv(
    csv_path: Path, rows: list[tuple[str, str, str, float, float]]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "pred_path", "gt_path", "psnr", "ssim"])
        writer.writerows(rows)


def print_progress(current: int, total: int, bar_width: int = 30) -> None:
    if total <= 0:
        return
    ratio = current / total
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\rProgress: [{bar}] {current}/{total} ({ratio * 100:6.2f}%)",
        end="",
        flush=True,
    )


def finish_progress() -> None:
    print()


def format_summary(
    config: EvalConfig, matched_count: int, avg_psnr: float, avg_ssim: float
) -> None:
    print("-" * 48)
    print("Mode:           y-channel")
    print(f"Pred dir:       {config.pred_dir}")
    print(f"GT dir:         {config.gt_dir}")
    print(f"Matched pairs:  {matched_count}")
    print(f"Average PSNR:   {avg_psnr:.6f}")
    print(f"Average SSIM:   {avg_ssim:.6f}")


def main() -> int:
    config = build_config(parse_args())

    pred_images = collect_images(config.pred_dir, config.recursive)
    gt_images = collect_images(config.gt_dir, config.recursive)

    common_names = sorted(set(pred_images) & set(gt_images), key=numeric_sort_key)
    missing_in_gt = sorted(set(pred_images) - set(gt_images), key=numeric_sort_key)
    missing_in_pred = sorted(set(gt_images) - set(pred_images), key=numeric_sort_key)

    if config.limit is not None:
        if config.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        common_names = common_names[: config.limit]

    if not common_names:
        print("No matched image pairs were found.", file=sys.stderr)
        print(f"pred_dir: {config.pred_dir}", file=sys.stderr)
        print(f"gt_dir:   {config.gt_dir}", file=sys.stderr)
        return 1

    if missing_in_gt:
        print(
            f"Warning: {len(missing_in_gt)} predicted images do not have matching GT files.",
            file=sys.stderr,
        )
    if missing_in_pred:
        print(
            f"Warning: {len(missing_in_pred)} GT images do not have matching prediction files.",
            file=sys.stderr,
        )

    rows: list[tuple[str, str, str, float, float]] = []
    total = len(common_names)
    for index, name in enumerate(common_names, start=1):
        pred_path = pred_images[name]
        gt_path = gt_images[name]
        metrics = compute_metrics_y_channel(pred_path, gt_path)
        rows.append((name, str(pred_path), str(gt_path), metrics.psnr, metrics.ssim))
        if not config.quiet:
            print_progress(index, total)

    if not config.quiet:
        finish_progress()

    avg_psnr = math.fsum(psnr for _, _, _, psnr, _ in rows) / len(rows)
    avg_ssim = math.fsum(ssim for _, _, _, _, ssim in rows) / len(rows)
    format_summary(config, len(rows), avg_psnr, avg_ssim)

    save_csv(config.csv_path, rows)
    print(f"CSV saved to: {config.csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
