from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    window = window_2d.unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        channels = predictions.shape[1]
        window = _gaussian_window(
            window_size=self.window_size,
            sigma=self.sigma,
            channels=channels,
            device=predictions.device,
            dtype=predictions.dtype,
        )

        mu_x = F.conv2d(predictions, window, padding=self.window_size // 2, groups=channels)
        mu_y = F.conv2d(targets, window, padding=self.window_size // 2, groups=channels)

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(predictions * predictions, window, padding=self.window_size // 2, groups=channels)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(targets * targets, window, padding=self.window_size // 2, groups=channels)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(predictions * targets, window, padding=self.window_size // 2, groups=channels)
            - mu_xy
        )

        c1 = self.k1**2
        c2 = self.k2**2

        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / denominator.clamp_min(1e-8)
        return 1.0 - ssim_map.mean()


class SpotterReconstructionLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.6, ssim_weight: float = 0.4) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mse_value = F.mse_loss(predictions, targets)
        ssim_value = self.ssim_loss(predictions, targets)
        total_loss = self.mse_weight * mse_value + self.ssim_weight * ssim_value
        return total_loss, {
            "mse": mse_value.detach(),
            "ssim_loss": ssim_value.detach(),
        }
