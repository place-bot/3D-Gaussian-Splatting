import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
PAIR_DIR = FIGURES / "beamer_pairs"

FINAL_VARIANTS = {
    "Sequence_01": "enhanced_long_finetune",
    "Sequence_02": "partition_mid_finetune",
    "Sequence_03": "partition_best_finetune",
    "Sequence_04": "partition_best_finetune",
    "Sequence_05": "full_finetune",
}

SOURCE_VARIANTS = {
    "Sequence_01": "enhanced_long",
    "Sequence_02": "partition_mid",
    "Sequence_03": "partition_best",
    "Sequence_04": "partition_best",
    "Sequence_05": "full",
}

BASELINE_VARIANTS = {
    "Sequence_01": "enhanced_long",
    "Sequence_02": "partition_mid",
    "Sequence_03": "full",
    "Sequence_04": "optimized",
    "Sequence_05": "full",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metric_rows(sequence: str, variant: str) -> list[dict]:
    path = ROOT / "work" / sequence / variant / "output" / "metrics_table.csv"
    with path.open(encoding="utf-8") as handle:
        return [
            {"index": int(float(row["index"])), "psnr": float(row["psnr"]), "ssim": float(row["ssim"])}
            for row in csv.DictReader(handle)
        ]


def count_colmap(sequence: str, variant: str) -> tuple[int, int, int]:
    candidate = ROOT / "work" / sequence / variant
    source_variant = SOURCE_VARIANTS.get(sequence, variant)
    source = ROOT / "work" / sequence / source_variant
    dataset_root = candidate if (candidate / "undistorted" / "sparse_text" / "images.txt").exists() else source
    sparse_text = dataset_root / "undistorted" / "sparse_text"
    image_lines = [
        line for line in (sparse_text / "images.txt").read_text(errors="ignore").splitlines()
        if line and not line.startswith("#")
    ]
    point_lines = [
        line for line in (sparse_text / "points3D.txt").read_text(errors="ignore").splitlines()
        if line and not line.startswith("#")
    ]
    input_dir = dataset_root / "input"
    if not input_dir.exists():
        input_dir = dataset_root / "undistorted" / "images"
    input_count = sum(1 for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return input_count, len(image_lines) // 2, len(point_lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_tables() -> None:
    render_rows = []
    colmap_rows = []
    geometry_rows = []
    local_rows = []
    ablation_rows = []
    for sequence, variant in FINAL_VARIANTS.items():
        metrics = read_json(ROOT / "work" / sequence / variant / "output" / "metrics.json")
        rows = read_metric_rows(sequence, variant)
        input_count, registered, points = count_colmap(sequence, variant)
        geometry = read_json(ROOT / "work" / sequence / variant / "evaluation" / "geometry_metrics.json")
        local_path = ROOT / "work" / sequence / variant / "evaluation" / "local_accuracy_measurements.csv"
        with local_path.open(encoding="utf-8") as handle:
            local_items = list(csv.DictReader(handle))
        render_rows.append(
            {
                "sequence": sequence,
                "variant": variant,
                "image_count": metrics["image_count"],
                "gaussian_count": metrics["gaussian_count"],
                "mean_psnr": metrics["mean_psnr"],
                "median_psnr": float(np.median([row["psnr"] for row in rows])),
                "min_psnr": metrics["min_psnr"],
                "max_psnr": metrics["max_psnr"],
                "mean_ssim": metrics["mean_ssim"],
                "median_ssim": float(np.median([row["ssim"] for row in rows])),
                "psnr_over_20": sum(row["psnr"] >= 20 for row in rows),
                "psnr_over_25": sum(row["psnr"] >= 25 for row in rows),
                "psnr_over_28": sum(row["psnr"] >= 28 for row in rows),
            }
        )
        colmap_rows.append(
            {
                "sequence": sequence,
                "variant": variant,
                "input_images": input_count,
                "registered_images": registered,
                "registration_rate": registered / input_count if input_count else 0.0,
                "sparse_points": points,
            }
        )
        geometry_rows.append(
            {
                "sequence": sequence,
                "variant": variant,
                "mean_error_m": geometry["mean_error_m"],
                "median_error_m": geometry["median_error_m"],
                "rmse_m": geometry["rmse_m"],
                "p90_error_m": geometry["p90_error_m"],
                "ratio_below_20cm": geometry["ratio_below_20cm"],
                "ratio_below_10cm": geometry["ratio_below_10cm"],
                "completion_ratio_below_20cm": geometry["completion_ratio_below_20cm"],
            }
        )
        local_rows.append(
            {
                "sequence": sequence,
                "variant": variant,
                "mean_local_median_error_m": float(np.mean([float(row["median_error_m"]) for row in local_items])),
                "best_local_median_error_m": float(np.min([float(row["median_error_m"]) for row in local_items])),
                "mean_local_ratio_below_20cm": float(np.mean([float(row["ratio_below_20cm"]) for row in local_items])),
                "mean_local_ratio_below_10cm": float(np.mean([float(row["ratio_below_10cm"]) for row in local_items])),
            }
        )

        baseline = BASELINE_VARIANTS[sequence]
        base_metrics = read_json(ROOT / "work" / sequence / baseline / "output" / "metrics.json")
        ablation_rows.append(
            {
                "sequence": sequence,
                "baseline_variant": baseline,
                "final_variant": variant,
                "baseline_psnr": base_metrics["mean_psnr"],
                "final_psnr": metrics["mean_psnr"],
                "psnr_gain": metrics["mean_psnr"] - base_metrics["mean_psnr"],
                "baseline_ssim": base_metrics["mean_ssim"],
                "final_ssim": metrics["mean_ssim"],
                "ssim_gain": metrics["mean_ssim"] - base_metrics["mean_ssim"],
            }
        )

    advanced_rows = [
        {
            "method": "Controlled 3DGS finetune",
            "sequence": "Sequence_04",
            "variant": "partition_best_finetune",
            **read_json(ROOT / "work" / "Sequence_04" / "partition_best_finetune" / "output" / "metrics.json"),
        },
        {
            "method": "Mip Splatting official",
            "sequence": "Sequence_04",
            "variant": "mip_splatting",
            **read_json(ROOT / "work" / "Sequence_04" / "mip_splatting" / "evaluation" / "metrics.json"),
        },
        {
            "method": "2DGS official",
            "sequence": "Sequence_04",
            "variant": "twodgs",
            **read_json(ROOT / "work" / "Sequence_04" / "twodgs" / "evaluation" / "metrics.json"),
        },
        {
            "method": "Depth RegularizedGS",
            "sequence": "Sequence_04",
            "variant": "depth_regularized_full",
            **read_json(ROOT / "work" / "Sequence_04" / "depth_regularized_full" / "evaluation" / "metrics.json"),
        },
    ]
    write_csv(TABLES / "render_quality_summary.csv", render_rows)
    write_csv(TABLES / "colmap_summary.csv", colmap_rows)
    write_csv(TABLES / "geometry_summary.csv", geometry_rows)
    write_csv(TABLES / "local_accuracy_summary.csv", local_rows)
    write_csv(TABLES / "ablation_summary.csv", ablation_rows)
    write_csv(TABLES / "advanced_methods_summary.csv", advanced_rows)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_plots() -> None:
    render_rows = list(csv.DictReader((TABLES / "render_quality_summary.csv").open(encoding="utf-8")))
    labels = [row["sequence"].replace("Sequence_", "Seq ") for row in render_rows]
    psnr = [float(row["mean_psnr"]) for row in render_rows]
    ssim = [float(row["mean_ssim"]) for row in render_rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8.8, 4.2))
    ax1.bar(x - 0.18, psnr, width=0.36, color="#315f8c", label="PSNR")
    ax1.axhline(20, color="#777777", linestyle="--", linewidth=1)
    ax1.axhline(25, color="#555555", linestyle="--", linewidth=1)
    ax1.axhline(28, color="#222222", linestyle="--", linewidth=1)
    ax1.set_ylabel("PSNR dB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, ssim, color="#b24a35", marker="o", linewidth=2, label="SSIM")
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0.55, 0.95)
    ax1.set_title("Final render quality")
    save_plot(FIGURES / "final_render_quality.png")

    geometry_rows = list(csv.DictReader((TABLES / "geometry_summary.csv").open(encoding="utf-8")))
    med = [float(row["median_error_m"]) * 100 for row in geometry_rows]
    ratio20 = [float(row["ratio_below_20cm"]) for row in geometry_rows]
    fig, ax1 = plt.subplots(figsize=(8.8, 4.2))
    ax1.bar(x - 0.18, med, width=0.36, color="#5f7f4f", label="Median error")
    ax1.axhline(20, color="#222222", linestyle="--", linewidth=1)
    ax1.axhline(10, color="#777777", linestyle="--", linewidth=1)
    ax1.set_ylabel("Median geometry error cm")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, ratio20, color="#9c3f5a", marker="s", linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Ratio below 20 cm")
    ax1.set_title("Geometry accuracy after similarity alignment")
    save_plot(FIGURES / "final_geometry_accuracy.png")

    ablation_rows = list(csv.DictReader((TABLES / "ablation_summary.csv").open(encoding="utf-8")))
    base = [float(row["baseline_psnr"]) for row in ablation_rows]
    final = [float(row["final_psnr"]) for row in ablation_rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.bar(x - 0.18, base, width=0.36, color="#b7b7b7", label="Before")
    ax.bar(x + 0.18, final, width=0.36, color="#315f8c", label="After")
    ax.axhline(25, color="#222222", linestyle="--", linewidth=1)
    ax.set_ylabel("PSNR dB")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Ablation and tuning improvement")
    save_plot(FIGURES / "final_ablation_gain.png")

    adv = list(csv.DictReader((TABLES / "advanced_methods_summary.csv").open(encoding="utf-8")))
    adv_labels = [row["method"].replace(" official", "") for row in adv]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(np.arange(len(adv)), [float(row["mean_psnr"]) for row in adv], color=["#315f8c", "#7667a6", "#b25a3c", "#5f7f4f"])
    ax.axhline(25, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(28, color="#222222", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(adv)))
    ax.set_xticklabels(adv_labels, rotation=12, ha="right")
    ax.set_ylabel("PSNR dB")
    ax.set_title("Advanced method comparison on Sequence 04")
    save_plot(FIGURES / "advanced_methods_comparison.png")


def resize_keep(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)


def make_labeled_canvas(images: list[Image.Image], labels: list[str], output: Path, cell=(520, 680), cols=2) -> None:
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * cell[0], rows * (cell[1] + 42)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        thumb = resize_keep(image.convert("RGB"), cell[0], cell[1])
        x = col * cell[0] + (cell[0] - thumb.width) // 2
        y = row * (cell[1] + 42)
        canvas.paste(thumb, (x, y))
        draw.text((col * cell[0] + 12, y + thumb.height + 10), labels[idx], fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def make_image_assets() -> None:
    FIGURES.mkdir(exist_ok=True)
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    input_images = []
    input_labels = []
    for sequence in FINAL_VARIANTS:
        files = sorted((ROOT / "data" / sequence / "images").glob("*.jpg"))
        input_images.append(Image.open(files[len(files) // 2]))
        input_labels.append(sequence)
    make_labeled_canvas(input_images, input_labels, FIGURES / "final_input_samples.jpg", cell=(300, 420), cols=5)

    for sequence, variant in FINAL_VARIANTS.items():
        render_paths = sorted((ROOT / "work" / sequence / variant / "output" / "renders").glob("*.jpg"))
        comparison_paths = sorted((ROOT / "work" / sequence / variant / "output" / "comparisons").glob("*.jpg"))
        make_labeled_canvas(
            [Image.open(path) for path in render_paths[:6]],
            [path.stem for path in render_paths[:6]],
            FIGURES / f"{sequence.lower()}_{variant}_renders_large.jpg",
            cell=(360, 500),
            cols=3,
        )
        make_labeled_canvas(
            [Image.open(path) for path in comparison_paths[:4]],
            [path.stem for path in comparison_paths[:4]],
            FIGURES / f"{sequence.lower()}_{variant}_comparisons_large.jpg",
            cell=(760, 500),
            cols=1,
        )
        for idx, path in enumerate([comparison_paths[0], comparison_paths[len(comparison_paths) // 2]], start=1):
            target = PAIR_DIR / f"{sequence.lower()}_pair_{idx}.jpg"
            Image.open(path).convert("RGB").save(target, quality=95)

    make_advanced_pair(
        ROOT / "work" / "Sequence_04" / "mip_splatting" / "output" / "train" / "ours_12000" / "gt_1",
        ROOT / "work" / "Sequence_04" / "mip_splatting" / "output" / "train" / "ours_12000" / "test_preds_1",
        PAIR_DIR / "sequence04_mip_pair.jpg",
    )
    make_advanced_pair(
        ROOT / "work" / "Sequence_04" / "twodgs" / "output" / "train" / "ours_12000" / "gt",
        ROOT / "work" / "Sequence_04" / "twodgs" / "output" / "train" / "ours_12000" / "renders",
        PAIR_DIR / "sequence04_twodgs_pair.jpg",
    )
    make_advanced_pair(
        ROOT / "work" / "Sequence_04" / "depth_regularized_full" / "output" / "train" / "ours_12000" / "gt",
        ROOT / "work" / "Sequence_04" / "depth_regularized_full" / "output" / "train" / "ours_12000" / "renders",
        PAIR_DIR / "sequence04_depth_regularized_pair.jpg",
    )


def make_advanced_pair(gt_dir: Path, render_dir: Path, output: Path) -> None:
    render_files = sorted(render_dir.glob("*.png"))
    if not render_files:
        render_files = sorted(render_dir.glob("*.jpg"))
    render_path = render_files[len(render_files) // 2]
    gt_path = gt_dir / render_path.name
    gt = Image.open(gt_path).convert("RGB")
    render = Image.open(render_path).convert("RGB")
    height = min(gt.height, render.height)
    gt = resize_keep(gt, 740, height)
    render = resize_keep(render, 740, height)
    gap = Image.new("RGB", (20, max(gt.height, render.height)), "white")
    canvas = Image.new("RGB", (gt.width + gap.width + render.width, max(gt.height, render.height)), "white")
    canvas.paste(gt, (0, 0))
    canvas.paste(gap, (gt.width, 0))
    canvas.paste(render, (gt.width + gap.width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 2.4))
    ax.axis("off")
    steps = ["Images", "Selection", "COLMAP", "3DGS", "Advanced methods", "Evaluation"]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (xpos, step) in enumerate(zip(xs, steps)):
        ax.text(xpos, 0.55, step, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#f3f3f3", edgecolor="#222222"))
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.075, 0.55), xytext=(xpos + 0.075, 0.55),
                        arrowprops=dict(arrowstyle="->", linewidth=1.3))
    ax.text(0.5, 0.12, "Stage 2 indoor reconstruction pipeline", ha="center", fontsize=11)
    save_plot(FIGURES / "pipeline.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    make_tables()
    make_pipeline()
    make_plots()
    make_image_assets()


if __name__ == "__main__":
    main()
