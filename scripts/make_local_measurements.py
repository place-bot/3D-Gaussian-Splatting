import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_stage2_geometry import read_ply_xyz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, default=None)
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


def robust_height(points: np.ndarray) -> float:
    z0, z1 = np.quantile(points[:, 2], [0.05, 0.95])
    return float(z1 - z0)


def select_spatially_separated(candidates: list[dict], count: int) -> list[dict]:
    selected: list[dict] = []
    for candidate in candidates:
        if all(abs(candidate["grid_x"] - item["grid_x"]) + abs(candidate["grid_y"] - item["grid_y"]) >= 2 for item in selected):
            selected.append(candidate)
        if len(selected) == count:
            return selected
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == count:
            return selected
    return selected


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    work_dir = args.root / "work" / args.sequence / args.variant
    recon_path = work_dir / "output" / "model_final_supersplat.ply"
    gt_path = args.root / "data" / args.sequence / "gt" / "gt_pd.ply"
    metrics_path = work_dir / "evaluation" / "geometry_metrics.json"

    recon = apply_transform(read_ply_xyz(recon_path), json.loads(metrics_path.read_text(encoding="utf-8"))["transform"])
    gt = read_ply_xyz(gt_path)
    if recon.shape[0] > 160000:
        recon = recon[rng.choice(recon.shape[0], 160000, replace=False)]
    if gt.shape[0] > 220000:
        gt = gt[rng.choice(gt.shape[0], 220000, replace=False)]

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
            recon_mask = (recon[:, 0] >= x0) & (recon[:, 0] < x1) & (recon[:, 1] >= y0) & (recon[:, 1] < y1)
            gt_mask = (gt[:, 0] >= x0) & (gt[:, 0] < x1) & (gt[:, 1] >= y0) & (gt[:, 1] < y1)
            recon_count = int(recon_mask.sum())
            gt_count = int(gt_mask.sum())
            if recon_count < 400 or gt_count < 800:
                continue
            recon_height = robust_height(recon[recon_mask])
            gt_height = robust_height(gt[gt_mask])
            if recon_height < 0.15 or gt_height < 0.15:
                continue
            candidates.append(
                {
                    "grid_x": ix,
                    "grid_y": iy,
                    "reconstruction_m": recon_height,
                    "ground_truth_m": gt_height,
                    "absolute_error_m": abs(recon_height - gt_height),
                    "reconstruction_points": recon_count,
                    "ground_truth_points": gt_count,
                    "support_score": min(recon_count, gt_count),
                }
            )

    if len(candidates) < args.count:
        raise RuntimeError(f"Only {len(candidates)} supported local measurement cells were found")

    candidates.sort(key=lambda row: (-row["support_score"], row["grid_x"], row["grid_y"]))
    selected = select_spatially_separated(candidates, args.count)
    rows = []
    for index, item in enumerate(selected, start=1):
        row = dict(item)
        row["measurement_id"] = f"local_height_{index}"
        rows.append(row)

    fieldnames = [
        "measurement_id",
        "grid_x",
        "grid_y",
        "reconstruction_m",
        "ground_truth_m",
        "absolute_error_m",
        "reconstruction_points",
        "ground_truth_points",
        "support_score",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.candidate_output is not None:
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        with args.candidate_output.open("w", newline="", encoding="utf-8") as handle:
            candidate_fields = [name for name in fieldnames if name != "measurement_id"]
            writer = csv.DictWriter(handle, fieldnames=candidate_fields)
            writer.writeheader()
            writer.writerows(candidates)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
