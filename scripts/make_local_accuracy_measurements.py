import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_stage2_geometry import read_ply_xyz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-x", type=int, default=6)
    parser.add_argument("--grid-y", type=int, default=5)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=22011958)
    return parser.parse_args()


def apply_transform(points: np.ndarray, transform: dict) -> np.ndarray:
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)
    return scale * (points @ rotation.T) + translation


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    work_dir = args.root / "work" / args.sequence / args.variant
    recon_path = work_dir / "output" / "model_final_supersplat.ply"
    gt_path = args.root / "data" / args.sequence / "gt" / "gt_pd.ply"
    metrics_path = work_dir / "evaluation" / "geometry_metrics.json"
    transform = json.loads(metrics_path.read_text(encoding="utf-8"))["transform"]

    recon = apply_transform(read_ply_xyz(recon_path), transform)
    gt = read_ply_xyz(gt_path)
    if recon.shape[0] > 160000:
        recon = recon[rng.choice(recon.shape[0], 160000, replace=False)]
    if gt.shape[0] > 220000:
        gt = gt[rng.choice(gt.shape[0], 220000, replace=False)]

    tree = cKDTree(gt)
    distances, _ = tree.query(recon, k=1)

    recon_quantile = np.quantile(recon[:, :2], [0.10, 0.90], axis=0)
    gt_quantile = np.quantile(gt[:, :2], [0.10, 0.90], axis=0)
    lo = np.maximum(recon_quantile[0], gt_quantile[0])
    hi = np.minimum(recon_quantile[1], gt_quantile[1])
    if np.any(hi <= lo):
        raise RuntimeError("No overlapping xy region for local measurements")

    candidates = []
    for ix in range(args.grid_x):
        for iy in range(args.grid_y):
            x0 = lo[0] + (hi[0] - lo[0]) * ix / args.grid_x
            x1 = lo[0] + (hi[0] - lo[0]) * (ix + 1) / args.grid_x
            y0 = lo[1] + (hi[1] - lo[1]) * iy / args.grid_y
            y1 = lo[1] + (hi[1] - lo[1]) * (iy + 1) / args.grid_y
            mask = (recon[:, 0] >= x0) & (recon[:, 0] < x1) & (recon[:, 1] >= y0) & (recon[:, 1] < y1)
            count = int(mask.sum())
            if count < 300:
                continue
            local = distances[mask]
            candidates.append(
                {
                    "grid_x": ix,
                    "grid_y": iy,
                    "sample_count": count,
                    "mean_error_m": float(np.mean(local)),
                    "median_error_m": float(np.median(local)),
                    "p90_error_m": float(np.percentile(local, 90)),
                    "ratio_below_20cm": float(np.mean(local <= 0.20)),
                    "ratio_below_10cm": float(np.mean(local <= 0.10)),
                }
            )

    if len(candidates) < args.count:
        raise RuntimeError(f"Only {len(candidates)} supported local cells were found")
    candidates.sort(key=lambda row: (-row["sample_count"], row["grid_x"], row["grid_y"]))
    selected = []
    for candidate in candidates:
        if all(abs(candidate["grid_x"] - item["grid_x"]) + abs(candidate["grid_y"] - item["grid_y"]) >= 2 for item in selected):
            selected.append(candidate)
        if len(selected) == args.count:
            break
    for candidate in candidates:
        if len(selected) == args.count:
            break
        if candidate not in selected:
            selected.append(candidate)

    rows = []
    for index, item in enumerate(selected, start=1):
        row = {"measurement_id": f"local_accuracy_{index}", **item}
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
