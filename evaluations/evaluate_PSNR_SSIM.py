#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


SUPPORTED_EXTS = (".jpg", ".png", ".jpeg", ".bmp")
DEFAULT_RESULT_DIR = Path("./Final_Model/SPA-Data/result_SPA-Data")
DEFAULT_GT_DIR = Path("./Datasets/SPA-Data/test/norain")


@dataclass(frozen=True)
class EvalConfig:
    result_dir: Path
    gt_dir: Path
    eval_name: str | None
    comparison_csv: Path | None



@dataclass(frozen=True)
class EvalResult:
    eval_name: str
    result_dir: Path
    gt_dir: Path
    image_count: int
    psnr: float
    ssim: float


def parse_args() -> argparse.Namespace:
    default_comparison_csv = Path("eval_Rain200H_Rain200L/rain200h_comparison.csv")

    parser = argparse.ArgumentParser(
        description="Compute MATLAB-style Y-channel PSNR and SSIM for paired images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate_PSNR_SSIM.py\n"
            "  python evaluate_PSNR_SSIM.py --result-dir \"Visual Results/m4_FADformer/Results/Rain200H\" --eval-name FADformer\n"
            "  python evaluate_PSNR_SSIM.py --comparison-csv rain200h_comparison.csv"
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing restored/result images.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help="Directory containing ground-truth images.",
    )
    parser.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Name shown in the summary output. Defaults to the result directory name.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=default_comparison_csv,
        help="Optional CSV used to compare this run with a recorded baseline. Accepts both PSNR/SSIM and CSV_PSNR/CSV_SSIM headers.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    comparison_csv = args.comparison_csv
    if comparison_csv is not None:
        comparison_csv = comparison_csv.resolve()
        if not comparison_csv.exists():
            comparison_csv = None

    return EvalConfig(
        result_dir=args.result_dir.resolve(),
        gt_dir=args.gt_dir.resolve(),
        eval_name=args.eval_name,
        comparison_csv=comparison_csv,
    )


def collect_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {image_dir}")

    image_files = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(image_files, key=lambda path: path.name)


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def matlab_rgb_to_y_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    rgb = image[:, :, :3].astype(np.float64)
    y = 16.0 + (
        65.481 * rgb[:, :, 0] + 128.553 * rgb[:, :, 1] + 24.966 * rgb[:, :, 2]
    ) / 255.0
    y = np.clip(np.round(y), 0, 255)
    return y.astype(np.uint8)


def compute_psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    rmse = math.sqrt(np.mean(np.square(diff)))
    if rmse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / rmse)


def gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    window = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return window / np.sum(window)


def ssim_index(image_a: np.ndarray, image_b: np.ndarray) -> float:
    if image_a.shape != image_b.shape:
        raise ValueError(f"Image shape mismatch: {image_a.shape} vs {image_b.shape}")

    if image_a.ndim != 2:
        raise ValueError(f"SSIM expects grayscale images, got shape: {image_a.shape}")

    height, width = image_a.shape
    if height < 11 or width < 11:
        return float("-inf")

    k1 = 0.01
    k2 = 0.03
    dynamic_range = 255.0
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    window = gaussian_window(11, 1.5)
    image_a = image_a.astype(np.float64)
    image_b = image_b.astype(np.float64)

    mu_a = convolve2d(image_a, window, mode="valid")
    mu_b = convolve2d(image_b, window, mode="valid")
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = convolve2d(image_a * image_a, window, mode="valid") - mu_a_sq
    sigma_b_sq = convolve2d(image_b * image_b, window, mode="valid") - mu_b_sq
    sigma_ab = convolve2d(image_a * image_b, window, mode="valid") - mu_ab

    ssim_map = ((2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    )
    return float(np.mean(ssim_map))


def compute_metrics(result_image: np.ndarray, gt_image: np.ndarray) -> tuple[float, float]:
    if result_image.shape != gt_image.shape:
        raise ValueError(f"Image size mismatch: {result_image.shape} vs {gt_image.shape}")

    result_y = matlab_rgb_to_y_uint8(result_image)
    gt_y = matlab_rgb_to_y_uint8(gt_image)
    return compute_psnr(result_y, gt_y), ssim_index(result_y, gt_y)


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


def evaluate(result_dir: Path, gt_dir: Path, eval_name: str | None = None) -> EvalResult:
    result_files = collect_image_files(result_dir)
    gt_files = collect_image_files(gt_dir)

    if not result_files:
        raise ValueError(f"No result images found in: {result_dir}")
    if len(result_files) != len(gt_files):
        raise ValueError(
            f"Image count mismatch. results: {len(result_files)}, gt: {len(gt_files)}"
        )

    total_psnr = 0.0
    total_ssim = 0.0
    image_count = len(result_files)
    for index, (result_path, gt_path) in enumerate(zip(result_files, gt_files), start=1):
        result_image = read_image(result_path)
        gt_image = read_image(gt_path)
        psnr_value, ssim_value = compute_metrics(result_image, gt_image)
        total_psnr += psnr_value
        total_ssim += ssim_value
        print_progress(index, image_count)

    finish_progress()

    final_eval_name = eval_name or result_dir.name
    return EvalResult(
        eval_name=final_eval_name,
        result_dir=result_dir,
        gt_dir=gt_dir,
        image_count=image_count,
        psnr=total_psnr / image_count,
        ssim=total_ssim / image_count,
    )


def lookup_csv_row(comparison_csv: Path, eval_name: str) -> dict[str, str] | None:
    with comparison_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("Method") == eval_name:
                return row
    return None


def get_csv_metric(row: dict[str, str], *field_names: str) -> float:
    for field_name in field_names:
        value = row.get(field_name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing CSV metric fields: {field_names}")


def print_result(result: EvalResult) -> None:
    print(f"Evaluation: {result.eval_name}")
    print(f"Result path: {result.result_dir}")
    print(f"GT path: {result.gt_dir}")
    print(f"Image count: {result.image_count}")
    print(f"PSNR: {result.psnr:.6f} SSIM: {result.ssim:.6f}")


def print_csv_comparison(result: EvalResult, comparison_csv: Path) -> None:
    row = lookup_csv_row(comparison_csv, result.eval_name)
    if row is None:
        print(f"Comparison CSV row not found for Method={result.eval_name}")
        return

    csv_psnr = get_csv_metric(row, "PSNR", "CSV_PSNR")
    csv_ssim = get_csv_metric(row, "SSIM", "CSV_SSIM")
    print(
        "CSV compare: "
        f"PSNR diff={result.psnr - csv_psnr:.6f}, "
        f"SSIM diff={result.ssim - csv_ssim:.6f}"
    )


def main() -> int:
    config = build_config(parse_args())
    result = evaluate(
        result_dir=config.result_dir,
        gt_dir=config.gt_dir,
        eval_name=config.eval_name,
    )
    print_result(result)

    if config.comparison_csv is not None:
        print_csv_comparison(result, config.comparison_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
