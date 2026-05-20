"""EEG data augmentation utilities and ablation presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

AugmentationPreset = Literal["none", "basic", "full"]


@dataclass(frozen=True)
class AugmentationConfig:
    """Batch-level augmentation settings for EEG tensors."""

    gaussian_noise_std: float = 0.0
    temporal_jitter_max: int = 0
    channel_dropout_prob: float = 0.0
    amplitude_scale_min: float = 1.0
    amplitude_scale_max: float = 1.0
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0


@dataclass(frozen=True)
class MixupBatch:
    """Mixup labels and mixing coefficient for one batch."""

    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float


def augmentation_config_from_preset(preset: AugmentationPreset) -> AugmentationConfig:
    """Return standard augmentation presets for ablation studies."""
    key = preset.lower()
    if key == "none":
        return AugmentationConfig()
    if key == "basic":
        return AugmentationConfig(
            gaussian_noise_std=0.01,
            temporal_jitter_max=8,
            channel_dropout_prob=0.0,
            amplitude_scale_min=0.9,
            amplitude_scale_max=1.1,
            mixup_alpha=0.0,
            mixup_prob=0.0,
        )
    if key == "full":
        return AugmentationConfig(
            gaussian_noise_std=0.02,
            temporal_jitter_max=16,
            channel_dropout_prob=0.1,
            amplitude_scale_min=0.8,
            amplitude_scale_max=1.2,
            mixup_alpha=0.2,
            mixup_prob=0.5,
        )
    raise ValueError(f"Unknown augmentation preset: {preset}")


def _apply_amplitude_scaling(x: torch.Tensor, cfg: AugmentationConfig) -> torch.Tensor:
    if cfg.amplitude_scale_min == 1.0 and cfg.amplitude_scale_max == 1.0:
        return x
    if cfg.amplitude_scale_min <= 0 or cfg.amplitude_scale_max <= 0:
        raise ValueError("Amplitude scale bounds must be positive.")
    if cfg.amplitude_scale_min > cfg.amplitude_scale_max:
        raise ValueError("amplitude_scale_min must be <= amplitude_scale_max.")
    scales = torch.empty((x.size(0), 1, 1), device=x.device, dtype=x.dtype).uniform_(
        cfg.amplitude_scale_min, cfg.amplitude_scale_max
    )
    return x * scales


def _apply_channel_dropout(x: torch.Tensor, cfg: AugmentationConfig) -> torch.Tensor:
    if cfg.channel_dropout_prob <= 0.0:
        return x
    if not (0.0 <= cfg.channel_dropout_prob < 1.0):
        raise ValueError("channel_dropout_prob must be in [0, 1).")
    drop_mask = torch.rand((x.size(0), x.size(1), 1), device=x.device) < cfg.channel_dropout_prob
    return x.masked_fill(drop_mask, 0.0)


def _apply_gaussian_noise(x: torch.Tensor, cfg: AugmentationConfig) -> torch.Tensor:
    if cfg.gaussian_noise_std <= 0.0:
        return x
    if cfg.gaussian_noise_std < 0.0:
        raise ValueError("gaussian_noise_std must be >= 0.")
    sample_std = x.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    noise = torch.randn_like(x) * (cfg.gaussian_noise_std * sample_std)
    return x + noise


def _apply_temporal_jitter(x: torch.Tensor, cfg: AugmentationConfig) -> torch.Tensor:
    if cfg.temporal_jitter_max <= 0:
        return x
    if x.size(-1) < 2:
        return x

    max_jitter = min(int(cfg.temporal_jitter_max), int(x.size(-1) - 1))
    padded = F.pad(x, (max_jitter, max_jitter), mode="reflect")
    shifts = torch.randint(
        low=-max_jitter,
        high=max_jitter + 1,
        size=(x.size(0),),
        device=x.device,
    )

    base = torch.arange(x.size(-1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
    gather_idx = base + shifts.unsqueeze(1) + max_jitter
    gather_idx = gather_idx.unsqueeze(1).expand(-1, x.size(1), -1)
    return torch.gather(padded, dim=2, index=gather_idx)


def _sample_mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: AugmentationConfig,
) -> tuple[torch.Tensor, MixupBatch | None]:
    if cfg.mixup_alpha <= 0.0 or cfg.mixup_prob <= 0.0 or x.size(0) < 2:
        return x, None
    if torch.rand(1, device=x.device).item() >= cfg.mixup_prob:
        return x, None

    beta_dist = torch.distributions.Beta(
        concentration1=torch.tensor(cfg.mixup_alpha, device=x.device),
        concentration0=torch.tensor(cfg.mixup_alpha, device=x.device),
    )
    lam = float(beta_dist.sample().item())
    lam = max(lam, 1.0 - lam)

    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mix = MixupBatch(targets_a=y, targets_b=y[perm], lam=lam)
    return mixed_x, mix


def apply_training_augmentations(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: AugmentationConfig,
) -> tuple[torch.Tensor, MixupBatch | None]:
    """Apply train-time augmentations to a batch of EEG signals."""
    out = x
    out = _apply_temporal_jitter(out, cfg)
    out = _apply_amplitude_scaling(out, cfg)
    out = _apply_channel_dropout(out, cfg)
    out = _apply_gaussian_noise(out, cfg)
    out, mixup_info = _sample_mixup(out, y, cfg)
    return out, mixup_info

