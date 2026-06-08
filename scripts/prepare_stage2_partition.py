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
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_quality(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return {
        "laplacian_var": float(cv2.Laplacian(image, cv2.CV_64F).var()),
        "mean": float(image.mean()),
        "std": float(image.std()),
    }


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
    if args.start_index < 1 or args.end_index > len(files) or args.start_index > args.end_index:
        raise ValueError("Invalid one-based frame interval")

    selected = files[args.start_index - 1 : args.end_index : args.stride]
    input_dir = args.work_dir / "input"
    if input_dir.exists() and args.overwrite:
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for new_index, src in enumerate(selected, start=1):
        dst_name = f"frame_{new_index:05d}.jpg"
        dst = input_dir / dst_name
        copy_image(src, dst, args.max_width)
        rows.append(
            {
                "new_name": dst_name,
                "source_name": src.name,
                "source_index": files.index(src) + 1,
                **image_quality(src),
            }
        )

    with (args.work_dir / "frame_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sequence": args.sequence_dir.name,
        "mode": "partition",
        "source_image_count": len(files),
        "selected_image_count": len(selected),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "stride": args.stride,
        "max_width": args.max_width,
        "mean_laplacian_var": float(np.mean([row["laplacian_var"] for row in rows])),
        "mean_brightness": float(np.mean([row["mean"] for row in rows])),
        "mean_contrast": float(np.mean([row["std"] for row in rows])),
    }
    (args.work_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
