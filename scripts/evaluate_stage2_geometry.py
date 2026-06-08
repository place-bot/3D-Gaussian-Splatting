import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon-ply", type=Path, required=True)
    parser.add_argument("--gt-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-recon-points", type=int, default=60000)
    parser.add_argument("--max-gt-points", type=int, default=120000)
    parser.add_argument("--icp-iterations", type=int, default=25)
    parser.add_argument("--trim-quantile", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=22011958)
    return parser.parse_args()


def read_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            decoded = line.decode("ascii", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        vertex_count = None
        properties = []
        fmt = None
        for line in header_lines:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                parts = line.split()
                if len(parts) == 3:
                    properties.append((parts[1], parts[2]))
        if vertex_count is None:
            raise ValueError(f"No vertex count in PLY: {path}")
        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                values = handle.readline().decode("ascii", errors="ignore").split()
                rows.append([float(values[0]), float(values[1]), float(values[2])])
            return np.asarray(rows, dtype=np.float64)
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt}: {path}")

        dtype_fields = []
        type_map = {
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "uchar": "u1",
            "uint8": "u1",
            "char": "i1",
            "int": "<i4",
            "int32": "<i4",
            "uint": "<u4",
            "uint32": "<u4",
            "short": "<i2",
            "ushort": "<u2",
        }
        for index, (kind, name) in enumerate(properties):
            if kind not in type_map:
                raise ValueError(f"Unsupported property type {kind} in {path}")
            dtype_fields.append((name if name else f"field_{index}", type_map[kind]))
        dtype = np.dtype(dtype_fields)
        data = np.frombuffer(handle.read(dtype.itemsize * vertex_count), dtype=dtype, count=vertex_count)
        return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def umeyama_similarity(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values) / max(variance, 1e-12))
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def robust_initial_alignment(recon: np.ndarray, gt: np.ndarray):
    recon_center = np.median(recon, axis=0)
    gt_center = np.median(gt, axis=0)
    recon_radius = np.percentile(np.linalg.norm(recon - recon_center, axis=1), 90)
    gt_radius = np.percentile(np.linalg.norm(gt - gt_center, axis=1), 90)
    scale = float(gt_radius / max(recon_radius, 1e-12))
    rotation = np.eye(3)
    translation = gt_center - scale * recon_center
    return scale, rotation, translation


def run_icp(recon: np.ndarray, gt: np.ndarray, iterations: int, trim_quantile: float):
    total_scale, total_rotation, total_translation = robust_initial_alignment(recon, gt)
    transformed = apply_similarity(recon, total_scale, total_rotation, total_translation)
    tree = cKDTree(gt)
    for _ in range(iterations):
        distances, indices = tree.query(transformed, k=1)
        threshold = np.quantile(distances, trim_quantile)
        mask = distances <= threshold
        if int(mask.sum()) < 20:
            break
        step_scale, step_rotation, step_translation = umeyama_similarity(transformed[mask], gt[indices[mask]])
        transformed = apply_similarity(transformed, step_scale, step_rotation, step_translation)
        total_translation = step_scale * step_rotation @ total_translation + step_translation
        total_rotation = step_rotation @ total_rotation
        total_scale = step_scale * total_scale
    distances, _ = tree.query(transformed, k=1)
    return transformed, distances, {
        "scale": float(total_scale),
        "rotation": total_rotation.tolist(),
        "translation": total_translation.tolist(),
    }


def write_xyz_ply(path: Path, points: np.ndarray, distances: np.ndarray):
    clipped = np.clip(distances / 0.20, 0.0, 1.0)
    colors = np.stack([255 * clipped, 255 * (1.0 - clipped), np.zeros_like(clipped)], axis=1).astype(np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            handle.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def robust_span(points: np.ndarray, axis: int) -> float:
    lo, hi = np.percentile(points[:, axis], [5, 95])
    return float(hi - lo)


def measurement_pairs(recon: np.ndarray, gt: np.ndarray):
    values = []
    labels = [
        ("x_width_5_95", lambda p: robust_span(p, 0)),
        ("y_depth_5_95", lambda p: robust_span(p, 1)),
        ("z_height_5_95", lambda p: robust_span(p, 2)),
        ("xy_planar_diagonal", lambda p: float(np.hypot(robust_span(p, 0), robust_span(p, 1)))),
        ("xyz_spatial_diagonal", lambda p: float(np.sqrt(robust_span(p, 0) ** 2 + robust_span(p, 1) ** 2 + robust_span(p, 2) ** 2))),
    ]
    for label, func in labels:
        recon_value = func(recon)
        gt_value = func(gt)
        values.append({
            "measurement": label,
            "reconstruction_m": recon_value,
            "ground_truth_m": gt_value,
            "absolute_error_m": abs(recon_value - gt_value),
        })
    return values


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recon = sample_points(read_ply_xyz(args.recon_ply), args.max_recon_points, rng)
    gt = sample_points(read_ply_xyz(args.gt_ply), args.max_gt_points, rng)
    transformed, distances, transform = run_icp(recon, gt, args.icp_iterations, args.trim_quantile)
    gt_to_recon_distances, _ = cKDTree(transformed).query(gt, k=1)

    metrics = {
        "recon_points_used": int(recon.shape[0]),
        "gt_points_used": int(gt.shape[0]),
        "mean_error_m": float(np.mean(distances)),
        "median_error_m": float(np.median(distances)),
        "rmse_m": float(np.sqrt(np.mean(distances**2))),
        "p90_error_m": float(np.percentile(distances, 90)),
        "p95_error_m": float(np.percentile(distances, 95)),
        "ratio_below_20cm": float(np.mean(distances <= 0.20)),
        "ratio_below_10cm": float(np.mean(distances <= 0.10)),
        "completion_mean_gt_to_recon_m": float(np.mean(gt_to_recon_distances)),
        "completion_ratio_below_20cm": float(np.mean(gt_to_recon_distances <= 0.20)),
        "transform": transform,
    }

    (args.output_dir / "geometry_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "measurement_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = measurement_pairs(transformed, gt)
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_xyz_ply(args.output_dir / "aligned_reconstruction_error_colored.ply", transformed, distances)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
