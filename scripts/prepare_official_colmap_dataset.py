import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--undistorted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    images_src = args.undistorted_dir / "images"
    sparse_src = args.undistorted_dir / "sparse"
    if not images_src.exists():
        raise FileNotFoundError(images_src)
    if not sparse_src.exists():
        raise FileNotFoundError(sparse_src)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(images_src, args.output_dir / "images", args.overwrite)

    sparse_dst = args.output_dir / "sparse" / "0"
    if sparse_dst.exists() and args.overwrite:
        shutil.rmtree(sparse_dst)
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for name in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
        source = sparse_src / name
        if not source.exists():
            source = args.undistorted_dir / "sparse_text" / name
        if source.exists():
            shutil.copy2(source, sparse_dst / name)

    required = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [name for name in required if not (sparse_dst / name).exists()]
    if missing:
        raise FileNotFoundError("Missing COLMAP files: " + ", ".join(missing))

    print(f"Prepared official COLMAP dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
