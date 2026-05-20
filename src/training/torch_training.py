"""Generic PyTorch training loop with config-driven setup."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import (
    EEGNetConfig,
    build_cnn_transformer_eeg,
    build_eegnet_model,
    build_multiscale_eeg_cnn,
)
from src.models.fbcsp_baseline import BaselineMetrics, compute_baseline_metrics
from src.training.augmentations import (
    AugmentationConfig,
    AugmentationPreset,
    apply_training_augmentations,
    augmentation_config_from_preset,
)
from src.training.leakage_safe_splits import (
    PerformanceSummary,
    SplitIndices,
    make_subject_dependent_run_split,
    make_subject_independent_splits,
    summarize_fold_metrics,
)

SplitMode = Literal["subject_dependent_run", "subject_independent_groupkfold", "subject_independent_logo"]


@dataclass(frozen=True)
class SubjectSplitConfig:
    """Configuration for subject split strategy."""

    mode: SplitMode = "subject_independent_groupkfold"
    test_run_fraction: float = 0.2
    n_splits: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    """Configurable settings for training and evaluation."""

    model_name: str = "multiscale_eeg_cnn"
    task_type: str = "task_a_binary_lr"
    subject_split: SubjectSplitConfig = SubjectSplitConfig()
    batch_size: int = 64
    learning_rates: tuple[float, ...] = (1e-3, 3e-4)
    epochs: int = 80
    dropout: float = 0.4
    bandpass_range: tuple[float, float] = (8.0, 30.0)
    weight_decay: float = 1e-4
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    augmentation_preset: AugmentationPreset = "none"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-5
    num_workers: int = 0
    random_seed: int = 42
    device: str | None = None
    checkpoint_dir: str = "artifacts/checkpoints"
    save_best_checkpoint: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class TrainingFoldResult:
    """Training/evaluation result for one fold."""

    mode: SplitMode
    fold: int
    learning_rate: float
    train_idx: np.ndarray
    val_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: BaselineMetrics
    checkpoint_path: Path | None
    history: list[dict[str, float]]


@dataclass(frozen=True)
class TrainingProtocolResult:
    """Collection of fold results and aggregate summary."""

    protocol_name: str
    split_mode: SplitMode
    fold_results: list[TrainingFoldResult]
    summary: PerformanceSummary


@dataclass(frozen=True)
class TrainingExperimentResult:
    """Full experiment output for both dependent and independent protocols."""

    config: TrainingConfig
    subject_dependent: TrainingProtocolResult
    subject_independent: TrainingProtocolResult
    report_table: pd.DataFrame


@dataclass(frozen=True)
class AugmentationAblationResult:
    """Ablation results across augmentation presets."""

    experiments: dict[str, TrainingExperimentResult]
    summary_table: pd.DataFrame


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(
    model_name: str,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float,
    dropout: float,
) -> tuple[nn.Module, str]:
    key = model_name.strip().lower()
    if key in {"eegnet", "braindecode_eegnet"}:
        eeg_cfg = EEGNetConfig(
            dropout_prob=dropout,
            prefer_braindecode=True,
            allow_fallback_model=True,
        )
        model, _, model_impl_name = build_eegnet_model(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
            config=eeg_cfg,
        )
        return model, model_impl_name
    if key in {"multiscale_eeg_cnn", "multiscale_cnn", "final_cnn"}:
        model = build_multiscale_eeg_cnn(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            dropout_prob=dropout,
        )
        return model, "MultiScaleEEGCNN"
    if key in {"cnn_transformer", "cnn_transformer_eeg"}:
        model = build_cnn_transformer_eeg(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            dropout_prob=dropout,
            transformer_layers=1,
            transformer_heads=2,
            transformer_embed_dim=64,
        )
        return model, "CNNTransformerEEG"
    raise ValueError(
        "Unknown model_name. Supported: eegnet, multiscale_eeg_cnn, cnn_transformer."
    )


def _class_weights(y: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0.0, 1.0, counts)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_pred_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_parts.append(preds)
            y_true_parts.append(y_batch.cpu().numpy())
    y_true = np.concatenate(y_true_parts, axis=0)
    y_pred = np.concatenate(y_pred_parts, axis=0)
    return y_true, y_pred


def _train_one_model_for_lr(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    learning_rate: float,
    cfg: TrainingConfig,
    checkpoint_path: Path | None,
    device: torch.device,
    fold: int,
) -> tuple[dict[str, torch.Tensor], float, list[dict[str, float]], BaselineMetrics]:
    n_classes = int(np.max(y) + 1)
    train_loader, val_loader = _make_loaders(
        X=X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    loss_weights = None
    if cfg.use_class_weights:
        loss_weights = _class_weights(y[train_idx], n_classes=n_classes, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights,
        label_smoothing=float(cfg.label_smoothing),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_bal_acc = -np.inf
    patience = 0
    history: list[dict[str, float]] = []
    best_metrics = BaselineMetrics(accuracy=0.0, balanced_accuracy=0.0, macro_f1=0.0, confusion_matrix=np.zeros((1, 1)))

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            X_batch, mixup = apply_training_augmentations(
                x=X_batch,
                y=y_batch,
                cfg=cfg.augmentation,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            if mixup is None:
                loss = criterion(logits, y_batch)
            else:
                loss = (
                    mixup.lam * criterion(logits, mixup.targets_a)
                    + (1.0 - mixup.lam) * criterion(logits, mixup.targets_b)
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        y_val_true, y_val_pred = _predict(model, val_loader, device=device)
        val_metrics = compute_baseline_metrics(y_val_true, y_val_pred)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(avg_loss),
                "val_accuracy": float(val_metrics.accuracy),
                "val_balanced_accuracy": float(val_metrics.balanced_accuracy),
                "val_macro_f1": float(val_metrics.macro_f1),
            }
        )

        if cfg.verbose:
            print(
                f"[fold {fold}] epoch {epoch + 1}/{cfg.epochs} lr={learning_rate:g} "
                f"loss={avg_loss:.4f} val_bal_acc={val_metrics.balanced_accuracy:.4f}"
            )

        if val_metrics.balanced_accuracy > (best_bal_acc + cfg.early_stopping_min_delta):
            best_bal_acc = float(val_metrics.balanced_accuracy)
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            patience = 0
        else:
            patience += 1

        if patience >= cfg.early_stopping_patience:
            break

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "learning_rate": float(learning_rate),
                "best_val_balanced_accuracy": float(best_bal_acc),
                "history": history,
            },
            checkpoint_path,
        )

    return best_state, float(best_bal_acc), history, best_metrics


def train_torch_split(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    split: SplitIndices,
    config: TrainingConfig,
) -> TrainingFoldResult:
    """Train one split using AdamW and early stopping on validation balanced accuracy."""
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_channels, n_times).")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be 1D and aligned with X.")

    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    val_idx = np.asarray(split.test_idx, dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split train/test indices must be non-empty.")

    _set_seed(config.random_seed + split.fold)
    device = _resolve_device(config.device)
    n_chans = int(X.shape[1])
    n_times = int(X.shape[2])
    n_outputs = int(np.max(y) + 1)

    best_model_state: dict[str, torch.Tensor] | None = None
    best_lr = float(config.learning_rates[0])
    best_bal_acc = -np.inf
    best_history: list[dict[str, float]] = []
    best_checkpoint_path: Path | None = None

    model_impl_name = ""
    for lr in config.learning_rates:
        model, impl_name = _build_model(
            model_name=config.model_name,
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
            dropout=config.dropout,
        )
        model_impl_name = impl_name
        model = model.to(device)

        checkpoint_path = None
        if config.save_best_checkpoint:
            checkpoint_path = (
                Path(config.checkpoint_dir)
                / config.model_name
                / split.mode
                / f"fold_{split.fold}_lr_{lr:.0e}.pt"
            )

        state, bal_acc, history, _ = _train_one_model_for_lr(
            model=model,
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            learning_rate=float(lr),
            cfg=config,
            checkpoint_path=checkpoint_path,
            device=device,
            fold=split.fold,
        )

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_lr = float(lr)
            best_model_state = state
            best_history = history
            best_checkpoint_path = checkpoint_path

    if best_model_state is None:
        raise RuntimeError("Failed to train model for any configured learning rate.")

    model, _ = _build_model(
        model_name=config.model_name,
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        sfreq=sfreq,
        dropout=config.dropout,
    )
    model = model.to(device)
    model.load_state_dict(best_model_state)

    _, val_loader = _make_loaders(
        X=X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    y_true, y_pred = _predict(model=model, loader=val_loader, device=device)
    final_metrics = compute_baseline_metrics(y_true=y_true, y_pred=y_pred)

    # Overwrite with final best checkpoint (one per fold) for convenience.
    if config.save_best_checkpoint and best_checkpoint_path is not None:
        final_path = (
            Path(config.checkpoint_dir)
            / config.model_name
            / split.mode
            / f"fold_{split.fold}_best.pt"
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": config.model_name,
                "model_impl_name": model_impl_name,
                "task_type": config.task_type,
                "subject_split_mode": split.mode,
                "bandpass_range": list(config.bandpass_range),
                "dropout": float(config.dropout),
                "learning_rate": float(best_lr),
                "model_state_dict": best_model_state,
                "metrics": {
                    "accuracy": float(final_metrics.accuracy),
                    "balanced_accuracy": float(final_metrics.balanced_accuracy),
                    "macro_f1": float(final_metrics.macro_f1),
                },
                "history": best_history,
            },
            final_path,
        )
        best_checkpoint_path = final_path

    return TrainingFoldResult(
        mode=split.mode,
        fold=split.fold,
        learning_rate=best_lr,
        train_idx=train_idx,
        val_idx=val_idx,
        y_true=y_true,
        y_pred=y_pred,
        metrics=final_metrics,
        checkpoint_path=best_checkpoint_path,
        history=best_history,
    )


def train_torch_splits(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    splits: Sequence[SplitIndices],
    config: TrainingConfig,
    protocol_name: str,
) -> TrainingProtocolResult:
    """Train/evaluate over multiple predefined splits."""
    fold_results = [train_torch_split(X=X, y=y, sfreq=sfreq, split=split, config=config) for split in splits]
    fold_metrics = [
        {
            "accuracy": result.metrics.accuracy,
            "balanced_accuracy": result.metrics.balanced_accuracy,
            "macro_f1": result.metrics.macro_f1,
        }
        for result in fold_results
    ]
    summary = summarize_fold_metrics(protocol=protocol_name, fold_metrics=fold_metrics)
    return TrainingProtocolResult(
        protocol_name=protocol_name,
        split_mode=fold_results[0].mode,
        fold_results=fold_results,
        summary=summary,
    )


def run_training_experiment(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    metadata: pd.DataFrame,
    config: TrainingConfig,
) -> TrainingExperimentResult:
    """Run and report both subject-dependent and subject-independent performance."""
    dependent_split = make_subject_dependent_run_split(
        metadata=metadata,
        test_run_fraction=config.subject_split.test_run_fraction,
        random_state=config.random_seed,
        fold=0,
    )
    dependent_result = train_torch_splits(
        X=X,
        y=y,
        sfreq=sfreq,
        splits=[dependent_split],
        config=config,
        protocol_name="subject_dependent",
    )

    if config.subject_split.mode == "subject_independent_logo":
        independent_splits = make_subject_independent_splits(metadata=metadata, method="logo")
    else:
        independent_splits = make_subject_independent_splits(
            metadata=metadata,
            method="groupkfold",
            n_splits=config.subject_split.n_splits,
        )

    independent_result = train_torch_splits(
        X=X,
        y=y,
        sfreq=sfreq,
        splits=independent_splits,
        config=config,
        protocol_name="subject_independent",
    )

    report_table = pd.DataFrame(
        [
            {
                "protocol": dependent_result.summary.protocol,
                "fold_count": dependent_result.summary.fold_count,
                "accuracy_mean": dependent_result.summary.metrics_mean.get("accuracy", np.nan),
                "accuracy_std": dependent_result.summary.metrics_std.get("accuracy", np.nan),
                "balanced_accuracy_mean": dependent_result.summary.metrics_mean.get("balanced_accuracy", np.nan),
                "balanced_accuracy_std": dependent_result.summary.metrics_std.get("balanced_accuracy", np.nan),
                "macro_f1_mean": dependent_result.summary.metrics_mean.get("macro_f1", np.nan),
                "macro_f1_std": dependent_result.summary.metrics_std.get("macro_f1", np.nan),
            },
            {
                "protocol": independent_result.summary.protocol,
                "fold_count": independent_result.summary.fold_count,
                "accuracy_mean": independent_result.summary.metrics_mean.get("accuracy", np.nan),
                "accuracy_std": independent_result.summary.metrics_std.get("accuracy", np.nan),
                "balanced_accuracy_mean": independent_result.summary.metrics_mean.get("balanced_accuracy", np.nan),
                "balanced_accuracy_std": independent_result.summary.metrics_std.get("balanced_accuracy", np.nan),
                "macro_f1_mean": independent_result.summary.metrics_mean.get("macro_f1", np.nan),
                "macro_f1_std": independent_result.summary.metrics_std.get("macro_f1", np.nan),
            },
        ]
    )

    return TrainingExperimentResult(
        config=config,
        subject_dependent=dependent_result,
        subject_independent=independent_result,
        report_table=report_table,
    )


def _parse_subject_split(payload: dict[str, Any]) -> SubjectSplitConfig:
    split = payload.get("subject_split", {}) or {}
    return SubjectSplitConfig(
        mode=str(split.get("mode", SubjectSplitConfig.mode)),
        test_run_fraction=float(split.get("test_run_fraction", SubjectSplitConfig.test_run_fraction)),
        n_splits=int(split.get("n_splits", SubjectSplitConfig.n_splits)),
    )


def _parse_augmentation(payload: dict[str, Any]) -> tuple[AugmentationPreset, AugmentationConfig]:
    aug = payload.get("augmentation", {}) or {}
    preset = str(aug.get("preset", payload.get("augmentation_preset", "none"))).lower()
    if preset not in {"none", "basic", "full"}:
        raise ValueError("augmentation preset must be one of: none, basic, full.")
    base = augmentation_config_from_preset(preset)  # type: ignore[arg-type]

    return preset, AugmentationConfig(
        gaussian_noise_std=float(aug.get("gaussian_noise_std", base.gaussian_noise_std)),
        temporal_jitter_max=int(aug.get("temporal_jitter_max", base.temporal_jitter_max)),
        channel_dropout_prob=float(aug.get("channel_dropout_prob", base.channel_dropout_prob)),
        amplitude_scale_min=float(aug.get("amplitude_scale_min", base.amplitude_scale_min)),
        amplitude_scale_max=float(aug.get("amplitude_scale_max", base.amplitude_scale_max)),
        mixup_alpha=float(aug.get("mixup_alpha", base.mixup_alpha)),
        mixup_prob=float(aug.get("mixup_prob", base.mixup_prob)),
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load training config from YAML file."""
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    subject_split = _parse_subject_split(data)
    augmentation_preset, augmentation_cfg = _parse_augmentation(data)

    lr_values = data.get("learning_rates")
    if lr_values is None:
        single_lr = data.get("learning_rate")
        if single_lr is not None:
            learning_rates = (float(single_lr),)
        else:
            learning_rates = TrainingConfig.learning_rates
    else:
        learning_rates = tuple(float(v) for v in lr_values)
    if len(learning_rates) == 0:
        raise ValueError("Config must provide at least one learning rate.")

    bandpass = data.get("bandpass_range", list(TrainingConfig.bandpass_range))
    if not isinstance(bandpass, (list, tuple)) or len(bandpass) != 2:
        raise ValueError("bandpass_range must be a list/tuple with two values.")

    return TrainingConfig(
        model_name=str(data.get("model_name", TrainingConfig.model_name)),
        task_type=str(data.get("task_type", TrainingConfig.task_type)),
        subject_split=subject_split,
        batch_size=int(data.get("batch_size", TrainingConfig.batch_size)),
        learning_rates=learning_rates,
        epochs=int(data.get("epochs", TrainingConfig.epochs)),
        dropout=float(data.get("dropout", TrainingConfig.dropout)),
        bandpass_range=(float(bandpass[0]), float(bandpass[1])),
        weight_decay=float(data.get("weight_decay", TrainingConfig.weight_decay)),
        use_class_weights=bool(data.get("use_class_weights", TrainingConfig.use_class_weights)),
        label_smoothing=float(data.get("label_smoothing", TrainingConfig.label_smoothing)),
        augmentation_preset=augmentation_preset,
        augmentation=augmentation_cfg,
        early_stopping_patience=int(
            data.get("early_stopping_patience", TrainingConfig.early_stopping_patience)
        ),
        early_stopping_min_delta=float(
            data.get("early_stopping_min_delta", TrainingConfig.early_stopping_min_delta)
        ),
        num_workers=int(data.get("num_workers", TrainingConfig.num_workers)),
        random_seed=int(data.get("random_seed", TrainingConfig.random_seed)),
        device=data.get("device", TrainingConfig.device),
        checkpoint_dir=str(data.get("checkpoint_dir", TrainingConfig.checkpoint_dir)),
        save_best_checkpoint=bool(
            data.get("save_best_checkpoint", TrainingConfig.save_best_checkpoint)
        ),
        verbose=bool(data.get("verbose", TrainingConfig.verbose)),
    )


def run_augmentation_ablation(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    metadata: pd.DataFrame,
    base_config: TrainingConfig,
    presets: Sequence[AugmentationPreset] = ("none", "basic", "full"),
) -> AugmentationAblationResult:
    """Run augmentation ablation: no augmentation, basic, full."""
    experiments: dict[str, TrainingExperimentResult] = {}
    rows: list[dict[str, float | str]] = []

    for preset in presets:
        aug_cfg = augmentation_config_from_preset(preset)
        cfg = replace(base_config, augmentation_preset=preset, augmentation=aug_cfg)
        result = run_training_experiment(
            X=X,
            y=y,
            sfreq=sfreq,
            metadata=metadata,
            config=cfg,
        )
        experiments[preset] = result

        dep = result.subject_dependent.summary
        indep = result.subject_independent.summary
        rows.append(
            {
                "augmentation_preset": preset,
                "subject_dependent_balanced_accuracy": dep.metrics_mean.get("balanced_accuracy", np.nan),
                "subject_dependent_macro_f1": dep.metrics_mean.get("macro_f1", np.nan),
                "subject_independent_balanced_accuracy": indep.metrics_mean.get(
                    "balanced_accuracy", np.nan
                ),
                "subject_independent_macro_f1": indep.metrics_mean.get("macro_f1", np.nan),
            }
        )

    return AugmentationAblationResult(
        experiments=experiments,
        summary_table=pd.DataFrame(rows),
    )
