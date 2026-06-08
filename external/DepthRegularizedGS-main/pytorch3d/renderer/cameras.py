import torch


class _Transform:
    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix

    def inverse(self):
        return _Transform(torch.linalg.inv(self._matrix))

    def compose(self, other):
        return _Transform(self._matrix @ other._matrix)

    def get_matrix(self):
        return self._matrix


class SfMPerspectiveCameras:
    def __init__(self, R: torch.Tensor, T: torch.Tensor):
        self.R = R
        self.T = T

    def get_world_to_view_transform(self):
        batch = self.R.shape[0]
        matrix = torch.eye(4, dtype=self.R.dtype, device=self.R.device).repeat(batch, 1, 1)
        matrix[:, :3, :3] = self.R
        matrix[:, 3, :3] = self.T
        return _Transform(matrix)

