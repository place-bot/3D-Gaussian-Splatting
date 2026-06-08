import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_files = sorted(args.renders_dir.glob("*"))
    rows = []
    for render_path in render_files:
        if render_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        gt_path = args.gt_dir / render_path.name
        if not gt_path.exists():
            continue
        pred = load_rgb(render_path)
        target = load_rgb(gt_path)
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch for {render_path.name}")
        rows.append(
            {
                "index": len(rows),
                "name": render_path.name,
                "psnr": psnr(pred, target),
                "ssim": float(structural_similarity(target, pred, channel_axis=-1, data_range=1.0)),
            }
        )

    if not rows:
        raise RuntimeError("No matching images found")
    summary = {
        "image_count": len(rows),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "min_psnr": float(np.min([row["psnr"] for row in rows])),
        "max_psnr": float(np.max([row["psnr"] for row in rows])),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "metrics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
