from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.autoencoder_config import AutoencoderModelConfig


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(F.adaptive_avg_pool2d(inputs, 1))))
        max_out = self.fc2(self.relu(self.fc1(F.adaptive_max_pool2d(inputs, 1))))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        outputs = inputs * channel_attention

        avg_spatial = torch.mean(outputs, dim=1, keepdim=True)
        max_spatial, _ = torch.max(outputs, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(
            self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1))
        )
        return outputs * spatial_attention


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            CBAM(out_channels, reduction=reduction),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class AutoencoderSpotterModel(nn.Module):
    def __init__(self, config: AutoencoderModelConfig) -> None:
        super().__init__()
        widths = [config.base_channels, config.base_channels * 2, config.base_channels * 4]

        self.enc1 = EncoderBlock(config.in_channels, widths[0], config.attention_reduction)
        self.enc2 = EncoderBlock(widths[0], widths[1], config.attention_reduction)
        self.enc3 = EncoderBlock(widths[1], widths[2], config.attention_reduction)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(widths[2], widths[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[2]),
            nn.GELU(),
        )

        self.dec1 = DecoderBlock(widths[2], widths[1])
        self.dec2 = DecoderBlock(widths[1], widths[0])
        self.dec3 = DecoderBlock(widths[0], widths[0] // 2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(widths[0] // 2, config.in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_1 = self.enc1(inputs)
        encoded_2 = self.enc2(encoded_1)
        encoded_3 = self.enc3(encoded_2)

        bottleneck = self.bottleneck(encoded_3)

        decoded_1 = self.dec1(bottleneck)
        decoded_2 = self.dec2(decoded_1)
        decoded_3 = self.dec3(decoded_2)
        return self.final_conv(decoded_3)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
