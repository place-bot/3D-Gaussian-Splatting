import torch


def so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    skew = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    scale = theta / (2.0 * torch.sin(theta).clamp_min(1e-7))
    return skew * scale.unsqueeze(-1)


def so3_exp_map(log_rotation: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(log_rotation, dim=-1, keepdim=True).clamp_min(1e-7)
    axis = log_rotation / theta
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(log_rotation.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=log_rotation.device, dtype=log_rotation.dtype)
    eye = eye.expand(log_rotation.shape[:-1] + (3, 3))
    theta_m = theta.unsqueeze(-1)
    return eye + torch.sin(theta_m) * k + (1.0 - torch.cos(theta_m)) * (k @ k)


def so3_relative_angle(rotation_1: torch.Tensor, rotation_2: torch.Tensor) -> torch.Tensor:
    relative = rotation_1.transpose(-1, -2) @ rotation_2
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))

