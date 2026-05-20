"""Final multi-scale EEG CNN model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class MultiScaleEEGCNNConfig:
    """Configuration for the multi-scale EEG CNN."""

    n_chans: int = 64
    n_times: int = 480
    n_outputs: int = 2
    temporal_kernels: tuple[int, int, int] = (16, 32, 64)
    branch_filters: int = 8
    spatial_depth_multiplier: int = 2
    separable_kernel: int = 16
    pool1_size: int = 4
    pool2_size: int = 8
    dropout_prob: float = 0.4
    batch_norm_momentum: float = 0.01
    batch_norm_eps: float = 1e-3


class MultiScaleEEGCNN(nn.Module):
    """Multi-branch EEGNet/ShallowConvNet-style architecture.

    Expected input shape: (batch, channels, time), e.g. (batch, 64, 480).
    """

    def __init__(self, config: MultiScaleEEGCNNConfig | None = None) -> None:
        super().__init__()
        cfg = config or MultiScaleEEGCNNConfig()
        self.config = cfg
        self._validate_config(cfg)

        k_short, k_medium, k_long = cfg.temporal_kernels
        self.temporal_branches = nn.ModuleList(
            [
                self._temporal_branch(kernel_size=k_short, out_filters=cfg.branch_filters),
                self._temporal_branch(kernel_size=k_medium, out_filters=cfg.branch_filters),
                self._temporal_branch(kernel_size=k_long, out_filters=cfg.branch_filters),
            ]
        )

        branch_out = cfg.branch_filters * len(cfg.temporal_kernels)
        spatial_out = branch_out * cfg.spatial_depth_multiplier

        # Depthwise spatial filtering across all EEG channels.
        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=branch_out,
                out_channels=spatial_out,
                kernel_size=(cfg.n_chans, 1),
                groups=branch_out,
                bias=False,
            ),
            nn.BatchNorm2d(
                spatial_out,
                momentum=cfg.batch_norm_momentum,
                eps=cfg.batch_norm_eps,
            ),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool1_size)),
            nn.Dropout(p=cfg.dropout_prob),
        )

        # Separable temporal convolution.
        separable_padding = cfg.separable_kernel // 2
        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=spatial_out,
                out_channels=spatial_out,
                kernel_size=(1, cfg.separable_kernel),
                groups=spatial_out,
                padding=(0, separable_padding),
                bias=False,
            ),
            nn.Conv2d(
                in_channels=spatial_out,
                out_channels=spatial_out,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                spatial_out,
                momentum=cfg.batch_norm_momentum,
                eps=cfg.batch_norm_eps,
            ),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool2_size)),
            nn.Dropout(p=cfg.dropout_prob),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.n_chans, cfg.n_times, dtype=torch.float32)
            flat_features = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(flat_features, cfg.n_outputs)

    @staticmethod
    def _validate_config(cfg: MultiScaleEEGCNNConfig) -> None:
        if cfg.n_chans <= 0 or cfg.n_times <= 0 or cfg.n_outputs <= 0:
            raise ValueError("n_chans, n_times, and n_outputs must be positive.")
        if len(cfg.temporal_kernels) != 3:
            raise ValueError("temporal_kernels must have exactly three values (short, medium, long).")
        if any(kernel <= 0 for kernel in cfg.temporal_kernels):
            raise ValueError("All temporal kernels must be positive.")
        if not (0.25 <= cfg.dropout_prob <= 0.5):
            raise ValueError("dropout_prob must be in [0.25, 0.5].")
        if cfg.branch_filters <= 0 or cfg.spatial_depth_multiplier <= 0:
            raise ValueError("branch_filters and spatial_depth_multiplier must be positive.")
        if cfg.pool1_size <= 0 or cfg.pool2_size <= 0:
            raise ValueError("pool sizes must be positive.")
        if cfg.separable_kernel <= 0:
            raise ValueError("separable_kernel must be positive.")

    @staticmethod
    def _temporal_branch(kernel_size: int, out_filters: int) -> nn.Sequential:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_filters,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                bias=False,
            )
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, channels, time).")
        x = x.unsqueeze(1)  # (batch, 1, channels, time)

        branch_outputs = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.spatial_block(x)
        x = self.separable_block(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        return self.classifier(features)


def build_multiscale_eeg_cnn(
    n_outputs: int,
    n_chans: int = 64,
    n_times: int = 480,
    temporal_kernels: Sequence[int] = (16, 32, 64),
    dropout_prob: float = 0.4,
) -> MultiScaleEEGCNN:
    """Convenience builder for the final model."""
    cfg = MultiScaleEEGCNNConfig(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        temporal_kernels=tuple(int(k) for k in temporal_kernels),  # type: ignore[arg-type]
        dropout_prob=float(dropout_prob),
    )
    return MultiScaleEEGCNN(config=cfg)

