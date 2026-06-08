import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-count", type=int, default=80)
    parser.add_argument("--white-value", type=int, default=215)
    parser.add_argument("--white-saturation", type=int, default=70)
    parser.add_argument("--motion-threshold", type=int, default=35)
    return parser.parse_args()


def select_files(input_dir: Path, max_count: int) -> list[Path]:
    files = sorted(input_dir.glob("*.jpg"))
    if len(files) <= max_count:
        return files
    indices = np.linspace(0, len(files) - 1, max_count, dtype=int)
    return [files[int(index)] for index in indices]


def main():
    args = parse_args()
    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    files = select_files(args.input_dir, args.max_count)
    previous_gray = None
    rows = []
    for index, path in enumerate(files, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {path}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = ((hsv[:, :, 2] >= args.white_value) & (hsv[:, :, 1] <= args.white_saturation)).astype(np.uint8) * 255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            motion_mask = np.zeros_like(gray, dtype=np.uint8)
        else:
            diff = cv2.absdiff(gray, previous_gray)
            motion_mask = (diff >= args.motion_threshold).astype(np.uint8) * 255
        previous_gray = gray

        combined = cv2.morphologyEx(cv2.max(white_mask, motion_mask), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
        mask_path = mask_dir / f"{path.stem}_mask.png"
        overlay_path = overlay_dir / f"{path.stem}_overlay.jpg"
        cv2.imwrite(str(mask_path), combined)

        overlay = image.copy()
        overlay[combined > 0] = (0.35 * overlay[combined > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(overlay_path), overlay)
        rows.append(
            {
                "image": path.name,
                "masked_ratio": float(np.mean(combined > 0)),
                "white_ratio": float(np.mean(white_mask > 0)),
                "motion_ratio": float(np.mean(motion_mask > 0)),
            }
        )

    (args.output_dir / "mask_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"image_count": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
