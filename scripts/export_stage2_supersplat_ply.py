from pathlib import Path
import argparse

import numpy as np
import torch


SH_C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def make_header(vertex_count):
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields.extend([f"f_dc_{index}" for index in range(3)])
    fields.extend([f"f_rest_{index}" for index in range(45)])
    fields.append("opacity")
    fields.extend([f"scale_{index}" for index in range(3)])
    fields.extend([f"rot_{index}" for index in range(4)])
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {vertex_count}"]
    lines.extend([f"property float {field}" for field in fields])
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    means = to_numpy(checkpoint["means"])
    log_scales = to_numpy(checkpoint["log_scales"])
    quats = to_numpy(checkpoint["quats"])
    opacity_logits = to_numpy(checkpoint["opacity_logits"]).reshape(-1, 1)
    color_logits = to_numpy(checkpoint["color_logits"])

    normals = np.zeros_like(means, dtype=np.float32)
    colors = 1.0 / (1.0 + np.exp(-color_logits))
    f_dc = (colors - 0.5) / SH_C0
    f_rest = np.zeros((means.shape[0], 45), dtype=np.float32)
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)

    data = np.concatenate([means, normals, f_dc, f_rest, opacity_logits, log_scales, quats], axis=1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(make_header(data.shape[0]))
        data.tofile(handle)
    print(f"Exported {data.shape[0]} gaussians to {args.output}")


if __name__ == "__main__":
    main()
