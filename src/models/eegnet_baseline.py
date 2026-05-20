"""Deep learning EEGNet baseline for motor-imagery tasks."""

from __future__ import annotations

import copy
import inspect
import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .fbcsp_baseline import BaselineMetrics, compute_baseline_metrics

EEGNetBackend = Literal["braindecode", "fallback"]


@dataclass(frozen=True)
class EEGNetConfig:
    """Training/model configuration for EEGNet baseline."""

    batch_size: int = 64
    n_epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_prob: float = 0.25
    use_class_weights: bool = True
    early_stopping_patience: int = 12
    early_stopping_metric: Literal["balanced_accuracy", "macro_f1", "accuracy"] = "balanced_accuracy"
    min_improvement: float = 1e-5
    num_workers: int = 0
    random_seed: int = 42
    device: str | None = None
    prefer_braindecode: bool = True
    allow_fallback_model: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class EEGNetFoldResult:
    """Training/evaluation outcome for one split."""

    fold: int
    backend: EEGNetBackend
    model_name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: BaselineMetrics
    history: list[dict[str, float]]


class _FallbackEEGNet(nn.Module):
    """Compact EEGNet-like architecture for environments without Braindecode."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        dropout_prob: float = 0.25,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        depthwise_kernel_length: int = 16,
    ) -> None:
        super().__init__()
        F2 = F1 * D
        padding_time = kernel_length // 2
        padding_depth = depthwise_kernel_length // 2

        self.features = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, padding_time), bias=False),
            nn.BatchNorm2d(F1, eps=1e-3, momentum=0.01),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_prob),
            nn.Conv2d(
                F1 * D,
                F1 * D,
                kernel_size=(1, depthwise_kernel_length),
                groups=F1 * D,
                padding=(0, padding_depth),
                bias=False,
            ),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_prob),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times, dtype=torch.float32)
            flat_features = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(flat_features, n_outputs)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        x = self.features(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._forward_features(x))


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_Xy(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError("X must have shape (batch, channels, time).")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes for EEGNet training.")


def _class_weights_tensor(y_train: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    class_counts = np.where(class_counts == 0.0, 1.0, class_counts)
    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _prepare_braindecode_kwargs(
    model_cls: Any,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float,
    dropout_prob: float,
) -> dict[str, Any]:
    signature = inspect.signature(model_cls)
    params = signature.parameters

    kwargs: dict[str, Any] = {}
    if "n_chans" in params:
        kwargs["n_chans"] = n_chans
    if "in_chans" in params:
        kwargs["in_chans"] = n_chans
    if "n_outputs" in params:
        kwargs["n_outputs"] = n_outputs
    if "n_classes" in params:
        kwargs["n_classes"] = n_outputs
    if "n_times" in params:
        kwargs["n_times"] = n_times
    if "input_window_samples" in params:
        kwargs["input_window_samples"] = n_times
    if "sfreq" in params:
        kwargs["sfreq"] = sfreq
    if "drop_prob" in params:
        kwargs["drop_prob"] = dropout_prob
    if "add_log_softmax" in params:
        kwargs["add_log_softmax"] = False
    return kwargs


def build_eegnet_model(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float,
    config: EEGNetConfig | None = None,
) -> tuple[nn.Module, EEGNetBackend, str]:
    """Build EEGNet model preferring Braindecode implementation."""
    cfg = config or EEGNetConfig()
    if cfg.prefer_braindecode:
        try:
            from braindecode import models as bd_models  # type: ignore

            candidate_names = ("EEGNet", "EEGNetv4", "EEGNetv")
            for name in candidate_names:
                model_cls = getattr(bd_models, name, None)
                if model_cls is None:
                    continue
                kwargs = _prepare_braindecode_kwargs(
                    model_cls=model_cls,
                    n_chans=n_chans,
                    n_times=n_times,
                    n_outputs=n_outputs,
                    sfreq=sfreq,
                    dropout_prob=cfg.dropout_prob,
                )
                model = model_cls(**kwargs)
                return model, "braindecode", f"braindecode.{name}"
        except Exception:
            if not cfg.allow_fallback_model:
                raise

    if not cfg.allow_fallback_model:
        raise ImportError("Braindecode EEGNet unavailable and fallback disabled.")

    fallback = _FallbackEEGNet(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        dropout_prob=cfg.dropout_prob,
    )
    return fallback, "fallback", "_FallbackEEGNet"


def _make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(y_pred)
            trues.append(y_batch.cpu().numpy())

    y_true = np.concatenate(trues, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return y_true, y_pred


def train_eegnet_split(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    config: EEGNetConfig | None = None,
    fold: int = 0,
    label_order: Sequence[int] | None = None,
) -> EEGNetFoldResult:
    """Train and evaluate EEGNet on one split."""
    cfg = config or EEGNetConfig()
    _validate_Xy(X, y)
    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    test_idx_array = np.asarray(test_idx, dtype=np.int64)
    if len(train_idx_array) == 0 or len(test_idx_array) == 0:
        raise ValueError("train_idx and test_idx must both be non-empty.")

    _set_global_seed(cfg.random_seed + int(fold))
    device = _resolve_device(cfg.device)

    n_chans = int(X.shape[1])
    n_times = int(X.shape[2])
    n_outputs = int(np.max(y) + 1)
    model, backend, model_name = build_eegnet_model(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        sfreq=sfreq,
        config=cfg,
    )
    model = model.to(device)

    train_loader, test_loader = _make_loaders(
        X=X,
        y=y,
        train_idx=train_idx_array,
        test_idx=test_idx_array,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    class_weights = None
    if cfg.use_class_weights:
        class_weights = _class_weights_tensor(
            y_train=y[train_idx_array],
            n_classes=n_outputs,
            device=device,
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    metric_name = cfg.early_stopping_metric
    best_metric = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience_count = 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        y_val_true, y_val_pred = _evaluate_model(model=model, loader=test_loader, device=device)
        val_metrics = compute_baseline_metrics(
            y_true=y_val_true,
            y_pred=y_val_pred,
            labels=None if label_order is None else np.asarray(label_order, dtype=np.int64),
        )
        tracked_metric = float(getattr(val_metrics, metric_name))
        mean_train_loss = epoch_loss / max(len(train_loader), 1)

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": mean_train_loss,
                "val_accuracy": val_metrics.accuracy,
                "val_balanced_accuracy": val_metrics.balanced_accuracy,
                "val_macro_f1": val_metrics.macro_f1,
            }
        )

        if cfg.verbose:
            print(
                f"[fold {fold}] epoch {epoch + 1}/{cfg.n_epochs} "
                f"loss={mean_train_loss:.4f} val_bal_acc={val_metrics.balanced_accuracy:.4f} "
                f"val_macro_f1={val_metrics.macro_f1:.4f}"
            )

        if tracked_metric > (best_metric + cfg.min_improvement):
            best_metric = tracked_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = _evaluate_model(model=model, loader=test_loader, device=device)
    final_metrics = compute_baseline_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=None if label_order is None else np.asarray(label_order, dtype=np.int64),
    )

    return EEGNetFoldResult(
        fold=int(fold),
        backend=backend,
        model_name=model_name,
        train_idx=train_idx_array,
        test_idx=test_idx_array,
        y_true=y_true,
        y_pred=y_pred,
        metrics=final_metrics,
        history=history,
    )


def train_eegnet_splits(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    splits: Sequence[object],
    config: EEGNetConfig | None = None,
    label_order: Sequence[int] | None = None,
) -> list[EEGNetFoldResult]:
    """Train/evaluate EEGNet across multiple split objects."""
    results: list[EEGNetFoldResult] = []
    for idx, split in enumerate(splits):
        if not hasattr(split, "train_idx") or not hasattr(split, "test_idx"):
            raise ValueError("Each split must provide `train_idx` and `test_idx`.")
        fold = int(getattr(split, "fold", idx))
        results.append(
            train_eegnet_split(
                X=X,
                y=y,
                sfreq=sfreq,
                train_idx=getattr(split, "train_idx"),
                test_idx=getattr(split, "test_idx"),
                config=config,
                fold=fold,
                label_order=label_order,
            )
        )
    return results


def summarize_eegnet_results(results: Sequence[EEGNetFoldResult]) -> dict[str, float]:
    """Summarize scalar metrics over folds."""
    if len(results) == 0:
        raise ValueError("No results provided.")
    accuracy = np.array([r.metrics.accuracy for r in results], dtype=float)
    balanced_accuracy = np.array([r.metrics.balanced_accuracy for r in results], dtype=float)
    macro_f1 = np.array([r.metrics.macro_f1 for r in results], dtype=float)
    return {
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std(ddof=0)),
        "balanced_accuracy_mean": float(balanced_accuracy.mean()),
        "balanced_accuracy_std": float(balanced_accuracy.std(ddof=0)),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std(ddof=0)),
    }

