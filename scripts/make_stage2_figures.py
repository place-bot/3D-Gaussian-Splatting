import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

FINAL_VARIANTS = {
    "Sequence_01": "full",
    "Sequence_02": "partition_mid",
    "Sequence_03": "full",
    "Sequence_04": "optimized",
    "Sequence_05": "full",
}

ABLATION_VARIANTS = [
    ("Sequence_01", "full", "Seq01 full"),
    ("Sequence_01", "enhanced_long", "Seq01 selected"),
    ("Sequence_02", "full", "Seq02 full"),
    ("Sequence_02", "optimized", "Seq02 selected"),
    ("Sequence_02", "partition_mid", "Seq02 partition"),
    ("Sequence_04", "full", "Seq04 full"),
    ("Sequence_04", "optimized", "Seq04 selected"),
]


def ensure_dirs():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def count_colmap(sequence: str, variant: str):
    base = ROOT / "work" / sequence / variant / "undistorted" / "sparse_text"
    image_path = base / "images.txt"
    point_path = base / "points3D.txt"
    image_count = sum(1 for line in image_path.read_text(errors="ignore").splitlines() if line and not line.startswith("#")) // 2
    point_count = sum(1 for line in point_path.read_text(errors="ignore").splitlines() if line and not line.startswith("#"))
    input_count = len(list((ROOT / "work" / sequence / variant / "input").glob("*.jpg")))
    if input_count == 0:
        input_count = image_count
    return input_count, image_count, point_count


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(sequence: str, variant: str):
    metrics = read_json(ROOT / "work" / sequence / variant / "output" / "metrics.json")
    rows = []
    with (ROOT / "work" / sequence / variant / "output" / "metrics_table.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"index": int(float(row["index"])), "psnr": float(row["psnr"]), "ssim": float(row["ssim"])})
    metrics["psnr_over_20"] = int(sum(row["psnr"] >= 20.0 for row in rows))
    metrics["psnr_over_25"] = int(sum(row["psnr"] >= 25.0 for row in rows))
    metrics["psnr_over_28"] = int(sum(row["psnr"] >= 28.0 for row in rows))
    metrics["median_psnr"] = float(np.median([row["psnr"] for row in rows]))
    metrics["median_ssim"] = float(np.median([row["ssim"] for row in rows]))
    return metrics, rows


def make_pipeline():
    fig, ax = plt.subplots(figsize=(12.5, 2.7))
    ax.axis("off")
    steps = [
        "shared image sequence",
        "frame selection",
        "COLMAP SfM",
        "3DGS training",
        "PLY export",
        "PSNR SSIM geometry",
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (x, text) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f1f3f4", edgecolor="#222222"),
        )
        if i + 1 < len(xs):
            ax.annotate("", xy=(xs[i + 1] - 0.07, 0.55), xytext=(x + 0.07, 0.55), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(0.5, 0.12, "Stage 2 indoor reconstruction pipeline", ha="center", fontsize=11)
    fig.savefig(FIGURES / "pipeline.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def resize_keep(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(max_width / image.width, max_height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def make_grid(paths, output, columns, cell_size=(360, 480), labels=None):
    rows = int(np.ceil(len(paths) / columns))
    width = columns * cell_size[0]
    height = rows * (cell_size[1] + 34)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(paths):
        row = idx // columns
        col = idx % columns
        image = resize_keep(Image.open(path), cell_size[0], cell_size[1])
        x = col * cell_size[0] + (cell_size[0] - image.width) // 2
        y = row * (cell_size[1] + 34)
        canvas.paste(image, (x, y))
        if labels:
            draw.text((col * cell_size[0] + 8, y + image.height + 6), labels[idx], fill=(20, 20, 20))
    canvas.save(output, quality=95)


def make_input_samples():
    paths = []
    labels = []
    for sequence in FINAL_VARIANTS:
        images = sorted((ROOT / "data" / sequence / "images").glob("*.jpg"))
        index = len(images) // 2
        paths.append(images[index])
        labels.append(f"{sequence} frame {index + 1}")
    make_grid(paths, FIGURES / "input_samples_all_sequences.jpg", columns=5, cell_size=(220, 300), labels=labels)


def make_render_grids():
    for sequence, variant in FINAL_VARIANTS.items():
        render_paths = sorted((ROOT / "work" / sequence / variant / "output" / "renders").glob("*.jpg"))[:8]
        comparison_paths = sorted((ROOT / "work" / sequence / variant / "output" / "comparisons").glob("*.jpg"))[:6]
        make_grid(
            render_paths,
            FIGURES / f"{sequence.lower()}_{variant}_renders.jpg",
            columns=4,
            cell_size=(230, 310),
            labels=[p.stem.replace("render_", "view ") for p in render_paths],
        )
        make_grid(
            comparison_paths,
            FIGURES / f"{sequence.lower()}_{variant}_comparisons.jpg",
            columns=2,
            cell_size=(560, 380),
            labels=[p.stem.replace("comparison_", "view ") for p in comparison_paths],
        )


def write_summary_tables():
    render_rows = []
    geometry_rows = []
    colmap_rows = []
    ablation_rows = []
    local_rows = []
    for sequence, variant in FINAL_VARIANTS.items():
        input_count, registered, points = count_colmap(sequence, variant)
        metrics, _ = read_metrics(sequence, variant)
        geometry = read_json(ROOT / "work" / sequence / variant / "evaluation" / "geometry_metrics.json")
        render_rows.append(
            {
                "sequence": sequence,
                "variant": variant,
                "image_count": metrics["image_count"],
                "gaussian_count": metrics["gaussian_count"],
                "mean_psnr": metrics["mean_psnr"],
                "median_psnr": metrics["median_psnr"],
                "mean_ssim": metrics["mean_ssim"],
                "median_ssim": metrics["median_ssim"],
                "psnr_over_20": metrics["psnr_over_20"],
                "psnr_over_25": metrics["psnr_over_25"],
                "psnr_over_28": metrics["psnr_over_28"],
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
        local_path = ROOT / "work" / sequence / variant / "evaluation" / "local_accuracy_measurements.csv"
        local_median = []
        local_ratio20 = []
        local_ratio10 = []
        with local_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                local_median.append(float(row["median_error_m"]))
                local_ratio20.append(float(row["ratio_below_20cm"]))
                local_ratio10.append(float(row["ratio_below_10cm"]))
        local_rows.append(
            {
                "sequence": sequence,
                "variant": variant,
                "mean_local_median_error_m": float(np.mean(local_median)),
                "mean_local_ratio_below_20cm": float(np.mean(local_ratio20)),
                "mean_local_ratio_below_10cm": float(np.mean(local_ratio10)),
            }
        )
    for sequence, variant, label in ABLATION_VARIANTS:
        metrics_path = ROOT / "work" / sequence / variant / "output" / "metrics.json"
        if not metrics_path.exists():
            continue
        input_count, registered, points = count_colmap(sequence, variant)
        metrics, _ = read_metrics(sequence, variant)
        ablation_rows.append(
            {
                "label": label,
                "sequence": sequence,
                "variant": variant,
                "input_images": input_count,
                "registered_images": registered,
                "sparse_points": points,
                "image_count": metrics["image_count"],
                "gaussian_count": metrics["gaussian_count"],
                "mean_psnr": metrics["mean_psnr"],
                "mean_ssim": metrics["mean_ssim"],
                "psnr_over_20": metrics["psnr_over_20"],
                "psnr_over_25": metrics["psnr_over_25"],
            }
        )
    for path, rows in [
        (TABLES / "render_quality_summary.csv", render_rows),
        (TABLES / "geometry_summary.csv", geometry_rows),
        (TABLES / "colmap_summary.csv", colmap_rows),
        (TABLES / "ablation_summary.csv", ablation_rows),
        (TABLES / "local_accuracy_summary.csv", local_rows),
    ]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def make_colmap_summary():
    labels, input_counts, registered, points = [], [], [], []
    for sequence, variant in FINAL_VARIANTS.items():
        inp, reg, pts = count_colmap(sequence, variant)
        labels.append(f"{sequence[-2:]} {variant}")
        input_counts.append(inp)
        registered.append(reg)
        points.append(pts)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(10.5, 4.4))
    ax1.bar(x - 0.18, input_counts, width=0.36, label="input", color="#c8d5e8")
    ax1.bar(x + 0.18, registered, width=0.36, label="registered", color="#31689b")
    ax1.set_ylabel("image count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, np.array(points) / 1000.0, marker="o", color="#9b2f2f", linewidth=2, label="sparse points in thousand")
    ax2.set_ylabel("sparse points in thousand")
    ax2.legend(loc="upper right")
    ax1.set_title("COLMAP registration and sparse points")
    fig.tight_layout()
    fig.savefig(FIGURES / "colmap_all_sequences.png", dpi=240)
    plt.close(fig)


def make_render_metric_summary():
    rows = []
    for sequence, variant in FINAL_VARIANTS.items():
        metrics, _ = read_metrics(sequence, variant)
        rows.append((f"{sequence[-2:]} {variant}", metrics))
    labels = [row[0] for row in rows]
    psnr = [row[1]["mean_psnr"] for row in rows]
    ssim = [row[1]["mean_ssim"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(labels, psnr, color="#31689b")
    axes[0].axhline(20.0, color="#9b2f2f", linestyle="--", label="20 dB")
    axes[0].axhline(25.0, color="#548235", linestyle="--", label="25 dB")
    axes[0].axhline(28.0, color="#c9a227", linestyle="--", label="28 dB")
    axes[0].set_ylabel("mean PSNR dB")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, ssim, color="#70ad47")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("mean SSIM")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Rendering quality of final models")
    fig.tight_layout()
    fig.savefig(FIGURES / "render_metrics_all_sequences.png", dpi=240)
    plt.close(fig)


def make_psnr_tier_summary():
    labels, over20, over25, over28 = [], [], [], []
    for sequence, variant in FINAL_VARIANTS.items():
        metrics, _ = read_metrics(sequence, variant)
        n = metrics["image_count"]
        labels.append(f"{sequence[-2:]} {variant}")
        over20.append(metrics["psnr_over_20"] / n)
        over25.append(metrics["psnr_over_25"] / n)
        over28.append(metrics["psnr_over_28"] / n)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(x - 0.22, over20, width=0.22, label="PSNR at least 20")
    ax.bar(x, over25, width=0.22, label="PSNR at least 25")
    ax.bar(x + 0.22, over28, width=0.22, label="PSNR at least 28")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("view ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.set_title("PSNR tier ratio")
    fig.tight_layout()
    fig.savefig(FIGURES / "psnr_tiers_all_sequences.png", dpi=240)
    plt.close(fig)


def make_ablation_metrics():
    labels, psnr, registered = [], [], []
    for sequence, variant, label in ABLATION_VARIANTS:
        if not (ROOT / "work" / sequence / variant / "output" / "metrics.json").exists():
            continue
        metrics, _ = read_metrics(sequence, variant)
        _, reg, _ = count_colmap(sequence, variant)
        labels.append(label)
        psnr.append(metrics["mean_psnr"])
        registered.append(reg)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(12, 4.4))
    ax1.bar(x, psnr, color="#31689b", label="mean PSNR")
    ax1.axhline(20.0, color="#9b2f2f", linestyle="--")
    ax1.set_ylabel("PSNR dB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, registered, marker="o", color="#548235", label="registered images")
    ax2.set_ylabel("registered images")
    lines = ax1.get_lines() + ax2.get_lines()
    labels_legend = [line.get_label() for line in lines]
    if labels_legend:
        ax1.legend(lines, labels_legend, loc="upper right")
    ax1.set_title("Ablation of data selection and partitioning")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_render_quality.png", dpi=240)
    plt.close(fig)


def make_geometry_summary():
    labels, mean_err, median_err, ratio20, ratio10 = [], [], [], [], []
    for sequence, variant in FINAL_VARIANTS.items():
        metrics = read_json(ROOT / "work" / sequence / variant / "evaluation" / "geometry_metrics.json")
        labels.append(f"{sequence[-2:]} {variant}")
        mean_err.append(metrics["mean_error_m"])
        median_err.append(metrics["median_error_m"])
        ratio20.append(metrics["ratio_below_20cm"])
        ratio10.append(metrics["ratio_below_10cm"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, mean_err, width=0.36, label="mean", color="#31689b")
    axes[0].bar(x + 0.18, median_err, width=0.36, label="median", color="#70ad47")
    axes[0].axhline(0.20, color="#9b2f2f", linestyle="--", label="20 cm")
    axes[0].axhline(0.10, color="#c9a227", linestyle="--", label="10 cm")
    axes[0].set_ylabel("distance m")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    axes[1].bar(x - 0.18, ratio20, width=0.36, label="under 20 cm", color="#31689b")
    axes[1].bar(x + 0.18, ratio10, width=0.36, label="under 10 cm", color="#70ad47")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(fontsize=8)
    fig.suptitle("Geometry accuracy after similarity alignment")
    fig.tight_layout()
    fig.savefig(FIGURES / "geometry_accuracy_all_sequences.png", dpi=240)
    plt.close(fig)


def make_local_accuracy_summary():
    labels, local_median, local_mean_ratio20, local_mean_ratio10 = [], [], [], []
    for sequence, variant in FINAL_VARIANTS.items():
        path = ROOT / "work" / sequence / variant / "evaluation" / "local_accuracy_measurements.csv"
        medians, r20, r10 = [], [], []
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                medians.append(float(row["median_error_m"]))
                r20.append(float(row["ratio_below_20cm"]))
                r10.append(float(row["ratio_below_10cm"]))
        labels.append(f"{sequence[-2:]} {variant}")
        local_median.append(float(np.mean(medians)))
        local_mean_ratio20.append(float(np.mean(r20)))
        local_mean_ratio10.append(float(np.mean(r10)))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(labels, local_median, color="#31689b")
    axes[0].axhline(0.20, color="#9b2f2f", linestyle="--")
    axes[0].axhline(0.10, color="#c9a227", linestyle="--")
    axes[0].set_ylabel("mean of local median errors m")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(x - 0.18, local_mean_ratio20, width=0.36, label="under 20 cm", color="#31689b")
    axes[1].bar(x + 0.18, local_mean_ratio10, width=0.36, label="under 10 cm", color="#70ad47")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("mean local ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(fontsize=8)
    fig.suptitle("Five local surface accuracy measurements per final model")
    fig.tight_layout()
    fig.savefig(FIGURES / "local_accuracy_all_sequences.png", dpi=240)
    plt.close(fig)


def make_training_curves():
    entries = [
        ("Sequence_05", "full", "Seq05 full"),
        ("Sequence_02", "partition_mid", "Seq02 partition"),
        ("Sequence_04", "optimized", "Seq04 selected"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for sequence, variant, label in entries:
        rows = []
        with (ROOT / "work" / sequence / variant / "output" / "training_log.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({key: float(value) for key, value in row.items()})
        iterations = np.array([row["iteration"] for row in rows])
        loss = np.array([row["loss"] for row in rows])
        psnr = np.array([row["psnr"] for row in rows])
        axes[0].plot(iterations, loss, label=label)
        axes[1].plot(iterations, psnr, label=label)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("Loss")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("sample PSNR dB")
    axes[1].set_title("Sample PSNR")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Training logs")
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves_selected.png", dpi=240)
    plt.close(fig)


def make_metrics_histograms():
    entries = [
        ("Sequence_02", "partition_mid", "Seq02 partition"),
        ("Sequence_05", "full", "Seq05 full"),
    ]
    fig, axes = plt.subplots(len(entries), 2, figsize=(10.5, 7.2))
    for row_index, (sequence, variant, label) in enumerate(entries):
        _, rows = read_metrics(sequence, variant)
        psnr = np.array([row["psnr"] for row in rows])
        ssim = np.array([row["ssim"] for row in rows])
        axes[row_index, 0].hist(psnr, bins=18, color="#31689b", edgecolor="white")
        axes[row_index, 0].axvline(20.0, color="#9b2f2f", linestyle="--")
        axes[row_index, 0].axvline(25.0, color="#548235", linestyle="--")
        axes[row_index, 0].set_title(f"{label} PSNR")
        axes[row_index, 1].hist(ssim, bins=18, color="#70ad47", edgecolor="white")
        axes[row_index, 1].set_title(f"{label} SSIM")
    fig.tight_layout()
    fig.savefig(FIGURES / "metrics_histograms_selected.png", dpi=240)
    plt.close(fig)


def make_error_topviews():
    for sequence, variant in [("Sequence_02", "partition_mid"), ("Sequence_05", "full")]:
        path = ROOT / "work" / sequence / variant / "evaluation" / "aligned_reconstruction_error_colored.ply"
        xs, ys, colors = [], [], []
        with path.open("r", encoding="ascii", errors="ignore") as handle:
            header = True
            for line in handle:
                if header:
                    if line.strip() == "end_header":
                        header = False
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
                colors.append([int(parts[3]) / 255.0, int(parts[4]) / 255.0, int(parts[5]) / 255.0])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        colors = np.asarray(colors)
        if xs.shape[0] > 70000:
            rng = np.random.default_rng(22011958)
            ids = rng.choice(xs.shape[0], 70000, replace=False)
            xs, ys, colors = xs[ids], ys[ids], colors[ids]
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.scatter(xs, ys, c=colors, s=1.0, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")
        ax.set_title(f"{sequence} {variant} aligned error")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{sequence.lower()}_{variant}_error_topview.png", dpi=240)
        plt.close(fig)


def make_mask_examples():
    paths = sorted((ROOT / "work" / "Sequence_05" / "full" / "masks" / "overlays").glob("*.jpg"))[:6]
    if paths:
        make_grid(
            paths,
            FIGURES / "photometric_mask_examples.jpg",
            columns=3,
            cell_size=(300, 405),
            labels=[p.stem.replace("_overlay", "") for p in paths],
        )


def main():
    ensure_dirs()
    make_pipeline()
    make_input_samples()
    make_render_grids()
    write_summary_tables()
    make_colmap_summary()
    make_render_metric_summary()
    make_psnr_tier_summary()
    make_ablation_metrics()
    make_geometry_summary()
    make_local_accuracy_summary()
    make_training_curves()
    make_metrics_histograms()
    make_error_topviews()
    make_mask_examples()


if __name__ == "__main__":
    main()
