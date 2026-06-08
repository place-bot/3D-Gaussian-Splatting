import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["uniform", "sharp"], default="sharp")
    parser.add_argument("--target-count", type=int, default=140)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_quality(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    laplacian_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
    mean = float(image.mean())
    std = float(image.std())
    # Penalize nearly saturated or nearly dark frames while preserving sharpness as the main signal.
    exposure_penalty = abs(mean - 128.0) * 0.25 + max(0.0, 35.0 - std) * 2.0
    score = laplacian_var - exposure_penalty
    return {
        "laplacian_var": laplacian_var,
        "mean": mean,
        "std": std,
        "score": score,
    }


def choose_uniform(files: list[Path], target_count: int) -> list[Path]:
    if len(files) <= target_count:
        return files
    indices = np.linspace(0, len(files) - 1, target_count, dtype=int)
    return [files[int(index)] for index in indices]


def choose_sharp(files: list[Path], target_count: int, metrics: dict[Path, dict]) -> list[Path]:
    if len(files) <= target_count:
        return files
    bins = np.array_split(np.arange(len(files)), target_count)
    chosen = []
    for bin_indices in bins:
        candidates = [files[int(index)] for index in bin_indices]
        best = max(candidates, key=lambda item: metrics[item]["score"])
        chosen.append(best)
    return chosen


def copy_image(src: Path, dst: Path, max_width: int):
    with Image.open(src) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            scale = max_width / image.width
            height = int(round(image.height * scale))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        image.save(dst, quality=95)


def main():
    args = parse_args()
    image_dir = args.sequence_dir / "images"
    files = sorted(image_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No jpg images found in {image_dir}")

    input_dir = args.work_dir / "input"
    if input_dir.exists() and args.overwrite:
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    metrics = {path: image_quality(path) for path in files}
    if args.mode == "uniform":
        selected = choose_uniform(files, args.target_count)
    else:
        selected = choose_sharp(files, args.target_count, metrics)

    rows = []
    for new_index, src in enumerate(selected, start=1):
        dst_name = f"frame_{new_index:05d}.jpg"
        dst = input_dir / dst_name
        copy_image(src, dst, args.max_width)
        row = {
            "new_name": dst_name,
            "source_name": src.name,
            "source_index": files.index(src),
            **metrics[src],
        }
        rows.append(row)

    with (args.work_dir / "frame_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sequence": args.sequence_dir.name,
        "mode": args.mode,
        "source_image_count": len(files),
        "selected_image_count": len(selected),
        "target_count": args.target_count,
        "max_width": args.max_width,
        "mean_laplacian_var": float(np.mean([metrics[path]["laplacian_var"] for path in selected])),
        "mean_brightness": float(np.mean([metrics[path]["mean"] for path in selected])),
        "mean_contrast": float(np.mean([metrics[path]["std"] for path in selected])),
    }
    (args.work_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
