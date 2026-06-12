import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

from gsplat import rasterization


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--seed", type=int, default=22011958)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-render-count", type=int, default=12)
    parser.add_argument("--densify-until", type=int, default=3500)
    parser.add_argument("--densify-every", type=int, default=500)
    parser.add_argument("--densify-fraction", type=float, default=0.08)
    parser.add_argument("--densify-max-new", type=int, default=1600)
    parser.add_argument("--lambda-ssim", type=float, default=0.0)
    parser.add_argument("--lambda-edge", type=float, default=0.0)
    parser.add_argument("--lambda-opacity", type=float, default=0.0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--allow-resume-resolution-mismatch", action="store_true")
    parser.add_argument("--lr-scale", type=float, default=1.0)
    return parser.parse_args()


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]
        cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images(path: Path) -> list[ImagePose]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    poses: list[ImagePose] = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        poses.append(ImagePose(image_id, qvec, tvec, camera_id, name))
    return sorted(poses, key=lambda item: item.name)


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
        colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32) / 255.0


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def camera_intrinsics(camera: Camera, scaled_width: int, scaled_height: int) -> np.ndarray:
    scale_x = scaled_width / camera.width
    scale_y = scaled_height / camera.height
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
        f, cx, cy = camera.params[:3]
        fx = f
        fy = f
    else:
        fx, fy, cx, cy = camera.params[:4]
    return np.array(
        [[fx * scale_x, 0.0, cx * scale_x], [0.0, fy * scale_y, cy * scale_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_images(image_dir: Path, poses: list[ImagePose], camera: Camera, max_side: int) -> tuple[torch.Tensor, int, int]:
    scale = min(1.0, max_side / max(camera.width, camera.height))
    width = max(16, int(round(camera.width * scale)))
    height = max(16, int(round(camera.height * scale)))
    loaded = []
    for pose in poses:
        with Image.open(image_dir / pose.name) as image:
            image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            loaded.append(np.asarray(image, dtype=np.float32) / 255.0)
    tensor = torch.from_numpy(np.stack(loaded, axis=0))
    return tensor, width, height


def build_viewmats_and_intrinsics(
    poses: list[ImagePose],
    cameras: dict[int, Camera],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    viewmats = []
    intrinsics = []
    for pose in poses:
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = qvec_to_rotmat(pose.qvec)
        viewmat[:3, 3] = pose.tvec.astype(np.float32)
        viewmats.append(viewmat)
        intrinsics.append(camera_intrinsics(cameras[pose.camera_id], width, height))
    return np.stack(viewmats, axis=0), np.stack(intrinsics, axis=0)


def filter_points(points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center[None, :], axis=1)
    threshold = np.quantile(distances, 0.985)
    keep = distances <= threshold
    return points[keep], colors[keep]


def nearest_neighbor_scales(points: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(points.astype(np.float32))
    values = []
    block = 1024
    for start in range(0, tensor.shape[0], block):
        part = tensor[start : start + block]
        dist = torch.cdist(part, tensor)
        row_ids = torch.arange(part.shape[0]) + start
        dist[torch.arange(part.shape[0]), row_ids] = float("inf")
        values.append(dist.min(dim=1).values)
    nn = torch.cat(values).numpy()
    lo = np.quantile(nn, 0.05)
    hi = np.quantile(nn, 0.95)
    nn = np.clip(nn, lo, hi)
    return np.maximum(nn * 0.85, 1e-4).astype(np.float32)


def logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


def make_optimizer(
    means: torch.nn.Parameter,
    log_scales: torch.nn.Parameter,
    quats: torch.nn.Parameter,
    opacity_logits: torch.nn.Parameter,
    color_logits: torch.nn.Parameter,
    background_logits: torch.nn.Parameter,
    scene_extent: float,
    lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": [means], "lr": 0.00025 * scene_extent * lr_scale},
            {"params": [log_scales], "lr": 0.006 * lr_scale},
            {"params": [quats], "lr": 0.001 * lr_scale},
            {"params": [opacity_logits], "lr": 0.05 * lr_scale},
            {"params": [color_logits], "lr": 0.015 * lr_scale},
            {"params": [background_logits], "lr": 0.01 * lr_scale},
        ],
        eps=1e-15,
    )


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def torch_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pred_nchw = pred.permute(2, 0, 1).unsqueeze(0)
    target_nchw = target.permute(2, 0, 1).unsqueeze(0)
    padding = window_size // 2
    mu_x = F.avg_pool2d(pred_nchw, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target_nchw, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred_nchw * pred_nchw, window_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target_nchw * target_nchw, window_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred_nchw * target_nchw, window_size, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_nchw = pred.permute(2, 0, 1).unsqueeze(0)
    target_nchw = target.permute(2, 0, 1).unsqueeze(0)
    channels = pred_nchw.shape[1]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3)
    weight_x = sobel_x.repeat(channels, 1, 1, 1)
    weight_y = sobel_y.repeat(channels, 1, 1, 1)
    pred_x = F.conv2d(pred_nchw, weight_x, padding=1, groups=channels)
    pred_y = F.conv2d(pred_nchw, weight_y, padding=1, groups=channels)
    target_x = F.conv2d(target_nchw, weight_x, padding=1, groups=channels)
    target_y = F.conv2d(target_nchw, weight_y, padding=1, groups=channels)
    return (pred_x - target_x).abs().mean() + (pred_y - target_y).abs().mean()


def evaluate(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    opacity_logits: torch.Tensor,
    color_logits: torch.Tensor,
    background_logits: torch.Tensor,
    images: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    indices: list[int],
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    metrics = []
    renders = []
    with torch.no_grad():
        normalized_quats = F.normalize(quats, dim=1)
        scales = log_scales.exp()
        opacities = opacity_logits.sigmoid().squeeze(-1)
        colors = color_logits.sigmoid()
        background = background_logits.sigmoid()
        for idx in indices:
            rendered, _, _ = rasterization(
                means,
                normalized_quats,
                scales,
                opacities,
                colors,
                viewmats[idx : idx + 1],
                intrinsics[idx : idx + 1],
                width,
                height,
                backgrounds=background,
            )
            pred = rendered[0].clamp(0.0, 1.0)
            target = images[idx]
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            item = {
                "index": float(idx),
                "psnr": psnr(pred, target),
                "ssim": float(structural_similarity(target_np, pred_np, channel_axis=-1, data_range=1.0)),
            }
            metrics.append(item)
            renders.append(pred_np)
    return metrics, renders


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8)).save(path, quality=95)


def save_comparison(path: Path, target: np.ndarray, render: np.ndarray) -> None:
    gap = np.ones((target.shape[0], 12, 3), dtype=np.float32)
    canvas = np.concatenate([target, gap, render], axis=1)
    save_image(path, canvas)


def densify_parameters(
    means: torch.nn.Parameter,
    log_scales: torch.nn.Parameter,
    quats: torch.nn.Parameter,
    opacity_logits: torch.nn.Parameter,
    color_logits: torch.nn.Parameter,
    fraction: float,
    max_new: int,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    if means.grad is None:
        return means, log_scales, quats, opacity_logits, color_logits
    with torch.no_grad():
        grad_norm = means.grad.norm(dim=1)
        count = max(256, int(means.shape[0] * fraction))
        count = min(count, means.shape[0], max_new)
        selected = torch.topk(grad_norm, k=count).indices
        jitter = torch.randn_like(means.data[selected]) * log_scales.data[selected].exp() * 0.35
        new_means = means.data[selected] + jitter
        new_log_scales = log_scales.data[selected] + math.log(0.82)
        new_quats = quats.data[selected]
        new_opacity_logits = opacity_logits.data[selected] - 0.2
        new_color_logits = color_logits.data[selected]
        means_data = torch.cat([means.data, new_means], dim=0)
        scales_data = torch.cat([log_scales.data, new_log_scales], dim=0)
        quats_data = torch.cat([quats.data, new_quats], dim=0)
        opacity_data = torch.cat([opacity_logits.data, new_opacity_logits], dim=0)
        color_data = torch.cat([color_logits.data, new_color_logits], dim=0)
    return (
        torch.nn.Parameter(means_data),
        torch.nn.Parameter(scales_data),
        torch.nn.Parameter(quats_data),
        torch.nn.Parameter(opacity_data),
        torch.nn.Parameter(color_data),
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sparse_dir = args.scene_dir / "sparse_text"
    image_dir = args.scene_dir / "images"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras(sparse_dir / "cameras.txt")
    poses = read_images(sparse_dir / "images.txt")
    points, point_colors = read_points(sparse_dir / "points3D.txt")
    points, point_colors = filter_points(points, point_colors)

    camera = cameras[poses[0].camera_id]
    images_cpu, width, height = load_images(image_dir, poses, camera, args.max_side)
    viewmats_np, intrinsics_np = build_viewmats_and_intrinsics(poses, cameras, width, height)

    device = torch.device("cuda")
    images = images_cpu.to(device)
    viewmats = torch.from_numpy(viewmats_np).to(device)
    intrinsics = torch.from_numpy(intrinsics_np).to(device)

    scale_values = nearest_neighbor_scales(points)
    scene_extent = float(np.linalg.norm(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)))
    scene_extent = max(scene_extent, 1e-3)

    corner_pixels = torch.cat(
        [
            images_cpu[:, : max(1, height // 10), : max(1, width // 10)].reshape(-1, 3),
            images_cpu[:, -max(1, height // 10) :, -max(1, width // 10) :].reshape(-1, 3),
        ],
        dim=0,
    )
    base_iterations = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        checkpoint_width = int(checkpoint.get("width", width))
        checkpoint_height = int(checkpoint.get("height", height))
        if (
            not args.allow_resume_resolution_mismatch
            and (checkpoint_width != width or checkpoint_height != height)
        ):
            raise ValueError("Resume checkpoint resolution does not match current training resolution")
        if checkpoint_width != width or checkpoint_height != height:
            print(
                json.dumps(
                    {
                        "warning": "resume checkpoint resolution differs from current training resolution",
                        "checkpoint_width": checkpoint_width,
                        "checkpoint_height": checkpoint_height,
                        "current_width": width,
                        "current_height": height,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        means = torch.nn.Parameter(checkpoint["means"].to(device))
        log_scales = torch.nn.Parameter(checkpoint["log_scales"].to(device))
        quats = torch.nn.Parameter(checkpoint["quats"].to(device))
        opacity_logits = torch.nn.Parameter(checkpoint["opacity_logits"].to(device))
        color_logits = torch.nn.Parameter(checkpoint["color_logits"].to(device))
        background_logits = torch.nn.Parameter(checkpoint["background_logits"].to(device))
        base_iterations = int(checkpoint.get("iterations", 0))
    else:
        means = torch.nn.Parameter(torch.from_numpy(points).to(device))
        log_scales = torch.nn.Parameter(torch.log(torch.from_numpy(scale_values).to(device).view(-1, 1).repeat(1, 3)))
        quats_init = torch.zeros((points.shape[0], 4), dtype=torch.float32, device=device)
        quats_init[:, 0] = 1.0
        quats = torch.nn.Parameter(quats_init)
        opacity_logits = torch.nn.Parameter(logit(torch.full((points.shape[0], 1), 0.12, dtype=torch.float32, device=device)))
        color_logits = torch.nn.Parameter(logit(torch.from_numpy(point_colors).to(device)))
        background_logits = torch.nn.Parameter(logit(corner_pixels.mean(dim=0).to(device)))

    optimizer = make_optimizer(means, log_scales, quats, opacity_logits, color_logits, background_logits, scene_extent, args.lr_scale)

    log_path = args.output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "image_index", "loss", "l1", "mse", "ssim", "edge", "psnr", "gaussians"])
        writer.writeheader()

    for iteration in range(1, args.iterations + 1):
        image_index = random.randrange(images.shape[0])
        optimizer.zero_grad(set_to_none=True)
        normalized_quats = F.normalize(quats, dim=1)
        scales = log_scales.exp()
        opacities = opacity_logits.sigmoid().squeeze(-1)
        colors = color_logits.sigmoid()
        background = background_logits.sigmoid()
        rendered, _, _ = rasterization(
            means,
            normalized_quats,
            scales,
            opacities,
            colors,
            viewmats[image_index : image_index + 1],
            intrinsics[image_index : image_index + 1],
            width,
            height,
            backgrounds=background,
        )
        pred = rendered[0].clamp(0.0, 1.0)
        target = images[image_index]
        l1 = (pred - target).abs().mean()
        mse = F.mse_loss(pred, target)
        ssim_value = torch_ssim(pred, target) if args.lambda_ssim > 0 else pred.new_tensor(0.0)
        edge_value = edge_loss(pred, target) if args.lambda_edge > 0 else pred.new_tensor(0.0)
        opacity_value = opacities.mean() if args.lambda_opacity > 0 else pred.new_tensor(0.0)
        loss = (
            0.8 * l1
            + 0.2 * mse
            + args.lambda_ssim * (1.0 - ssim_value)
            + args.lambda_edge * edge_value
            + args.lambda_opacity * opacity_value
        )
        loss.backward()
        optimizer.step()

        if (
            iteration <= args.densify_until
            and iteration % args.densify_every == 0
            and iteration < args.iterations
        ):
            means, log_scales, quats, opacity_logits, color_logits = densify_parameters(
                means,
                log_scales,
                quats,
                opacity_logits,
                color_logits,
                args.densify_fraction,
                args.densify_max_new,
            )
            optimizer = make_optimizer(
                means,
                log_scales,
                quats,
                opacity_logits,
                color_logits,
                background_logits,
                scene_extent,
                args.lr_scale,
            )

        if iteration == 1 or iteration % args.eval_every == 0 or iteration == args.iterations:
            current_psnr = psnr(pred.detach(), target.detach())
            row = {
                "iteration": iteration,
                "image_index": image_index,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "ssim": float(ssim_value.detach().cpu()),
                "edge": float(edge_value.detach().cpu()),
                "psnr": current_psnr,
                "gaussians": int(means.shape[0]),
            }
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writerow(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    selected_indices = np.linspace(0, images.shape[0] - 1, args.save_render_count, dtype=int).tolist()
    all_indices = list(range(images.shape[0]))
    selected_metrics, selected_renders = evaluate(
        means,
        log_scales,
        quats,
        opacity_logits,
        color_logits,
        background_logits,
        images,
        viewmats,
        intrinsics,
        width,
        height,
        selected_indices,
    )
    all_metrics, _ = evaluate(
        means,
        log_scales,
        quats,
        opacity_logits,
        color_logits,
        background_logits,
        images,
        viewmats,
        intrinsics,
        width,
        height,
        all_indices,
    )

    render_dir = args.output_dir / "renders"
    comparison_dir = args.output_dir / "comparisons"
    for pose_index, render in zip(selected_indices, selected_renders, strict=True):
        target_np = images_cpu[pose_index].numpy()
        save_image(render_dir / f"render_{pose_index + 1:04d}.jpg", render)
        save_comparison(comparison_dir / f"comparison_{pose_index + 1:04d}.jpg", target_np, render)

    average_metrics = {
        "image_count": len(all_metrics),
        "selected_image_count": len(selected_metrics),
        "mean_psnr": float(np.mean([item["psnr"] for item in all_metrics])),
        "mean_ssim": float(np.mean([item["ssim"] for item in all_metrics])),
        "min_psnr": float(np.min([item["psnr"] for item in all_metrics])),
        "max_psnr": float(np.max([item["psnr"] for item in all_metrics])),
        "gaussian_count": int(means.shape[0]),
        "width": width,
        "height": height,
        "iterations": base_iterations + args.iterations,
        "additional_iterations": args.iterations,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(average_metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "metrics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(all_metrics)

    torch.save(
        {
            "means": means.detach().cpu(),
            "log_scales": log_scales.detach().cpu(),
            "quats": quats.detach().cpu(),
            "opacity_logits": opacity_logits.detach().cpu(),
            "color_logits": color_logits.detach().cpu(),
            "background_logits": background_logits.detach().cpu(),
            "width": width,
            "height": height,
            "iterations": base_iterations + args.iterations,
            "additional_iterations": args.iterations,
            "metrics": average_metrics,
        },
        args.output_dir / "model_final.pt",
    )
    print(json.dumps(average_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
