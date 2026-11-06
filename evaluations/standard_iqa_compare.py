import argparse
import csv
import json
from pathlib import Path

import pyiqa
import torch
from tqdm import tqdm


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute standard BRISQUE/NIQE with pyiqa and compare to a legacy CSV."
    )
    parser.add_argument("--folder", type=Path, required=True, help="Image folder to evaluate.")
    parser.add_argument("--legacy-csv", type=Path, default=None, help="Legacy result CSV to compare.")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV path for pyiqa scores.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    return parser.parse_args()


def list_images(folder: Path):
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS]
    )


def to_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return float(value.reshape(-1)[0].item())
    return float(value)


def read_legacy_average(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    average_row = next((row for row in rows if row.get("") == "Average"), None)
    if average_row is None:
        raise ValueError(f"Average row not found in {csv_path}")
    return {
        "brisque": float(average_row["brisque"]),
        "niqe": float(average_row["niqe"]),
    }


def main():
    args = parse_args()
    folder = args.folder.resolve()
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")

    brisque_metric = pyiqa.create_metric("brisque", device=args.device)
    niqe_metric = pyiqa.create_metric("niqe", device=args.device)

    rows = []
    for image_path in tqdm(images, desc=f"Evaluating {folder.name}"):
        brisque_score = to_float(brisque_metric(str(image_path)))
        niqe_score = to_float(niqe_metric(str(image_path)))
        rows.append(
            {
                "image": image_path.name,
                "brisque": brisque_score,
                "niqe": niqe_score,
            }
        )

    brisque_avg = sum(row["brisque"] for row in rows) / len(rows)
    niqe_avg = sum(row["niqe"] for row in rows) / len(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "brisque", "niqe"])
        for row in rows:
            writer.writerow([row["image"], f"{row['brisque']:.12f}", f"{row['niqe']:.12f}"])
        writer.writerow(["Average", f"{brisque_avg:.12f}", f"{niqe_avg:.12f}"])

    summary = {
        "folder": str(folder),
        "count": len(rows),
        "pyiqa_average": {
            "brisque": brisque_avg,
            "niqe": niqe_avg,
        },
        "legacy_average": None,
        "delta_pyiqa_minus_legacy": None,
    }

    if args.legacy_csv is not None:
        legacy_average = read_legacy_average(args.legacy_csv.resolve())
        summary["legacy_average"] = legacy_average
        summary["delta_pyiqa_minus_legacy"] = {
            "brisque": brisque_avg - legacy_average["brisque"],
            "niqe": niqe_avg - legacy_average["niqe"],
        }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
