"""Optional CNN-Transformer variant built on top of multi-scale EEG CNN blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CNNTransformerConfig:
    """Configuration for optional CNN-Transformer model."""

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
    transformer_layers: int = 1
    transformer_heads: int = 2
    transformer_embed_dim: int = 64
    transformer_ff_multiplier: int = 2
    transformer_dropout: float = 0.1
    use_cls_token: bool = True


class CNNTransformerEEG(nn.Module):
    """CNN feature extractor + small Transformer encoder classifier.

    Input shape: (batch, channels, time), e.g. (batch, 64, 480).
    """

    def __init__(self, config: CNNTransformerConfig | None = None) -> None:
        super().__init__()
        cfg = config or CNNTransformerConfig()
        self.config = cfg
        self._validate_config(cfg)

        self.temporal_branches = nn.ModuleList(
            [
                self._temporal_branch(cfg.temporal_kernels[0], cfg.branch_filters),
                self._temporal_branch(cfg.temporal_kernels[1], cfg.branch_filters),
                self._temporal_branch(cfg.temporal_kernels[2], cfg.branch_filters),
            ]
        )

        branch_out = cfg.branch_filters * len(cfg.temporal_kernels)
        spatial_out = branch_out * cfg.spatial_depth_multiplier

        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=branch_out,
                out_channels=spatial_out,
                kernel_size=(cfg.n_chans, 1),
                groups=branch_out,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_out, momentum=cfg.batch_norm_momentum, eps=cfg.batch_norm_eps),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool1_size)),
            nn.Dropout(p=cfg.dropout_prob),
        )

        sep_pad = cfg.separable_kernel // 2
        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=spatial_out,
                out_channels=spatial_out,
                kernel_size=(1, cfg.separable_kernel),
                groups=spatial_out,
                padding=(0, sep_pad),
                bias=False,
            ),
            nn.Conv2d(spatial_out, spatial_out, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_out, momentum=cfg.batch_norm_momentum, eps=cfg.batch_norm_eps),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool2_size)),
            nn.Dropout(p=cfg.dropout_prob),
        )

        self.token_projection = nn.Linear(spatial_out, cfg.transformer_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_embed_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_embed_dim * cfg.transformer_ff_multiplier,
            dropout=cfg.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.transformer_layers,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.n_chans, cfg.n_times, dtype=torch.float32)
            token_count = self._cnn_feature_map(dummy).shape[-1]
        self.max_tokens = int(token_count + (1 if cfg.use_cls_token else 0))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.max_tokens, cfg.transformer_embed_dim)
        )
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.transformer_embed_dim))
        else:
            self.cls_token = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.transformer_embed_dim),
            nn.Dropout(cfg.dropout_prob),
            nn.Linear(cfg.transformer_embed_dim, cfg.n_outputs),
        )

    @staticmethod
    def _validate_config(cfg: CNNTransformerConfig) -> None:
        if cfg.n_chans <= 0 or cfg.n_times <= 0 or cfg.n_outputs <= 0:
            raise ValueError("n_chans, n_times, and n_outputs must be positive.")
        if len(cfg.temporal_kernels) != 3:
            raise ValueError("temporal_kernels must be exactly three values (short, medium, long).")
        if not (0.25 <= cfg.dropout_prob <= 0.5):
            raise ValueError("dropout_prob must be in [0.25, 0.5].")
        if cfg.transformer_layers not in (1, 2):
            raise ValueError("transformer_layers must be 1 or 2.")
        if cfg.transformer_heads not in (2, 3, 4):
            raise ValueError("transformer_heads must be 2, 3, or 4.")
        if cfg.transformer_embed_dim <= 0 or cfg.transformer_embed_dim > 128:
            raise ValueError("transformer_embed_dim must be in [1, 128].")
        if cfg.transformer_embed_dim % cfg.transformer_heads != 0:
            raise ValueError("transformer_embed_dim must be divisible by transformer_heads.")

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

    def _cnn_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, channels, time).")
        x = x.unsqueeze(1)  # (B, 1, C, T)
        branches = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(branches, dim=1)
        x = self.spatial_block(x)
        x = self.separable_block(x)
        x = x.squeeze(2)  # (B, C_feat, T_tokens)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._cnn_feature_map(x)  # (B, C_feat, T_tokens)
        tokens = fmap.transpose(1, 2)  # (B, T_tokens, C_feat)
        tokens = self.token_projection(tokens)  # (B, T_tokens, D)

        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        if tokens.size(1) > self.positional_embedding.size(1):
            raise ValueError(
                f"Token length {tokens.size(1)} exceeds positional embedding size "
                f"{self.positional_embedding.size(1)}."
            )

        tokens = tokens + self.positional_embedding[:, : tokens.size(1), :]
        encoded = self.transformer(tokens)

        if self.cls_token is not None:
            summary = encoded[:, 0, :]
        else:
            summary = encoded.mean(dim=1)
        return self.classifier(summary)


def build_cnn_transformer_eeg(
    n_outputs: int,
    n_chans: int = 64,
    n_times: int = 480,
    temporal_kernels: Sequence[int] = (16, 32, 64),
    transformer_layers: int = 1,
    transformer_heads: int = 2,
    transformer_embed_dim: int = 64,
    dropout_prob: float = 0.4,
) -> CNNTransformerEEG:
    """Convenience builder for optional CNN-Transformer model."""
    cfg = CNNTransformerConfig(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        temporal_kernels=tuple(int(k) for k in temporal_kernels),  # type: ignore[arg-type]
        transformer_layers=int(transformer_layers),
        transformer_heads=int(transformer_heads),
        transformer_embed_dim=int(transformer_embed_dim),
        dropout_prob=float(dropout_prob),
    )
    return CNNTransformerEEG(config=cfg)


def compare_cnn_vs_transformer(
    cnn_metrics: dict[str, float],
    transformer_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute metric deltas (transformer minus CNN baseline)."""
    keys = ("accuracy_mean", "balanced_accuracy_mean", "macro_f1_mean")
    deltas: dict[str, float] = {}
    for key in keys:
        if key in cnn_metrics and key in transformer_metrics:
            deltas[f"delta_{key}"] = float(transformer_metrics[key] - cnn_metrics[key])
    return deltas
