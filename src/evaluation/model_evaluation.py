"""Unified evaluation and reporting across model families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class ModelEvaluationReport:
    """Structured evaluation outputs for one model/protocol."""

    model_name: str
    class_names: tuple[str, ...]
    fold_metrics: pd.DataFrame
    summary_mean_std: pd.DataFrame
    per_class_f1_by_fold: pd.DataFrame
    per_class_f1_summary: pd.DataFrame
    confusion_matrix_sum: pd.DataFrame
    confusion_matrix_per_fold: dict[int, np.ndarray]
    subject_wise_performance: pd.DataFrame
    training_curves: pd.DataFrame


def _coerce_fold_results(result_or_results: Any) -> list[Any]:
    if hasattr(result_or_results, "fold_results"):
        return list(getattr(result_or_results, "fold_results"))
    if isinstance(result_or_results, list):
        return result_or_results
    if isinstance(result_or_results, tuple):
        return list(result_or_results)
    return [result_or_results]


def _extract_indices(fold_result: Any) -> np.ndarray:
    if hasattr(fold_result, "test_idx"):
        return np.asarray(getattr(fold_result, "test_idx"), dtype=np.int64)
    if hasattr(fold_result, "val_idx"):
        return np.asarray(getattr(fold_result, "val_idx"), dtype=np.int64)
    raise ValueError("Fold result must have either `test_idx` or `val_idx`.")


def _scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def _default_class_names(y_all: Sequence[int]) -> tuple[str, ...]:
    labels = sorted(set(int(v) for v in y_all))
    return tuple(f"class_{label}" for label in labels)


def _build_training_curve_table(fold_results: Sequence[Any]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold_result in fold_results:
        fold = int(getattr(fold_result, "fold", 0))
        history = getattr(fold_result, "history", None)
        if not history:
            continue
        for item in history:
            rows.append(
                {
                    "fold": fold,
                    "epoch": float(item.get("epoch", np.nan)),
                    "train_loss": float(item.get("train_loss", np.nan)),
                    "val_accuracy": float(item.get("val_accuracy", np.nan)),
                    "val_balanced_accuracy": float(item.get("val_balanced_accuracy", np.nan)),
                    "val_macro_f1": float(item.get("val_macro_f1", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _build_subject_wise_table(
    fold_results: Sequence[Any],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if "subject_id" not in metadata.columns:
        raise ValueError("metadata must contain `subject_id`.")

    rows: list[dict[str, float | int]] = []
    for fold_result in fold_results:
        fold = int(getattr(fold_result, "fold", 0))
        idx = _extract_indices(fold_result)
        y_true = np.asarray(getattr(fold_result, "y_true"))
        y_pred = np.asarray(getattr(fold_result, "y_pred"))
        if len(idx) != len(y_true) or len(y_true) != len(y_pred):
            raise ValueError(f"Fold {fold}: index/prediction lengths do not match.")

        fold_subjects = metadata.iloc[idx]["subject_id"].to_numpy(dtype=np.int64)
        for subject in np.unique(fold_subjects):
            mask = fold_subjects == subject
            acc, bal_acc, macro_f1 = _scalar_metrics(y_true[mask], y_pred[mask])
            rows.append(
                {
                    "fold": fold,
                    "subject_id": int(subject),
                    "n_samples": int(mask.sum()),
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "macro_f1": macro_f1,
                }
            )

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        return per_fold

    summary = (
        per_fold.groupby("subject_id", as_index=False)[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "subject_id",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
    ]
    return summary.sort_values("subject_id").reset_index(drop=True)


def evaluate_model_results(
    result_or_results: Any,
    metadata: pd.DataFrame,
    class_names: Sequence[str] | None = None,
    model_name: str = "model",
) -> ModelEvaluationReport:
    """Create full evaluation outputs for one model/protocol."""
    fold_results = _coerce_fold_results(result_or_results)
    if len(fold_results) == 0:
        raise ValueError("No fold results provided.")

    y_all: list[int] = []
    fold_rows: list[dict[str, float | int]] = []
    per_class_rows: list[dict[str, float | int | str]] = []
    confusion_per_fold: dict[int, np.ndarray] = {}
    confusion_sum: np.ndarray | None = None

    for fold_result in fold_results:
        fold = int(getattr(fold_result, "fold", 0))
        y_true = np.asarray(getattr(fold_result, "y_true"), dtype=np.int64)
        y_pred = np.asarray(getattr(fold_result, "y_pred"), dtype=np.int64)
        if len(y_true) != len(y_pred):
            raise ValueError(f"Fold {fold}: y_true/y_pred length mismatch.")

        y_all.extend(y_true.tolist())
        acc, bal_acc, macro_f1 = _scalar_metrics(y_true, y_pred)
        fold_rows.append(
            {
                "fold": fold,
                "n_samples": int(len(y_true)),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
            }
        )

        labels = np.unique(np.concatenate([y_true, y_pred]))
        class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        for label, value in zip(labels.tolist(), class_f1.tolist()):
            per_class_rows.append(
                {
                    "fold": fold,
                    "class_index": int(label),
                    "per_class_f1": float(value),
                }
            )

    resolved_class_names = tuple(class_names) if class_names is not None else _default_class_names(y_all)
    class_indices = np.arange(len(resolved_class_names), dtype=np.int64)

    for fold_result in fold_results:
        fold = int(getattr(fold_result, "fold", 0))
        y_true = np.asarray(getattr(fold_result, "y_true"), dtype=np.int64)
        y_pred = np.asarray(getattr(fold_result, "y_pred"), dtype=np.int64)
        cm = confusion_matrix(y_true, y_pred, labels=class_indices)
        confusion_per_fold[fold] = cm
        confusion_sum = cm if confusion_sum is None else (confusion_sum + cm)

    if confusion_sum is None:
        confusion_sum = np.zeros((len(class_indices), len(class_indices)), dtype=np.int64)

    fold_metrics_df = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    summary_df = pd.DataFrame(
        {
            "metric": ["accuracy", "balanced_accuracy", "macro_f1"],
            "mean": [
                float(fold_metrics_df["accuracy"].mean()),
                float(fold_metrics_df["balanced_accuracy"].mean()),
                float(fold_metrics_df["macro_f1"].mean()),
            ],
            "std": [
                float(fold_metrics_df["accuracy"].std(ddof=0)),
                float(fold_metrics_df["balanced_accuracy"].std(ddof=0)),
                float(fold_metrics_df["macro_f1"].std(ddof=0)),
            ],
        }
    )

    per_class_df = pd.DataFrame(per_class_rows)
    if per_class_df.empty:
        per_class_fold = pd.DataFrame(columns=["fold", "class_index", "class_name", "per_class_f1"])
        per_class_summary = pd.DataFrame(
            columns=["class_index", "class_name", "per_class_f1_mean", "per_class_f1_std"]
        )
    else:
        per_class_df["class_name"] = per_class_df["class_index"].apply(
            lambda idx: resolved_class_names[int(idx)]
            if int(idx) < len(resolved_class_names)
            else f"class_{int(idx)}"
        )
        per_class_fold = per_class_df.sort_values(["fold", "class_index"]).reset_index(drop=True)
        per_class_summary = (
            per_class_fold.groupby(["class_index", "class_name"], as_index=False)["per_class_f1"]
            .agg(["mean", "std"])
            .reset_index()
        )
        per_class_summary.columns = [
            "class_index",
            "class_name",
            "per_class_f1_mean",
            "per_class_f1_std",
        ]

    subject_table = _build_subject_wise_table(fold_results=fold_results, metadata=metadata)
    training_curves = _build_training_curve_table(fold_results=fold_results)

    confusion_df = pd.DataFrame(
        confusion_sum,
        index=[f"true_{name}" for name in resolved_class_names],
        columns=[f"pred_{name}" for name in resolved_class_names],
    )

    return ModelEvaluationReport(
        model_name=model_name,
        class_names=resolved_class_names,
        fold_metrics=fold_metrics_df,
        summary_mean_std=summary_df,
        per_class_f1_by_fold=per_class_fold,
        per_class_f1_summary=per_class_summary,
        confusion_matrix_sum=confusion_df,
        confusion_matrix_per_fold=confusion_per_fold,
        subject_wise_performance=subject_table,
        training_curves=training_curves,
    )


def plot_training_curves(
    report: ModelEvaluationReport,
    output_path: str | Path | None = None,
) -> plt.Figure | None:
    """Plot training curves from report history.

    Returns None when no training-curve history exists.
    """
    curves = report.training_curves
    if curves.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for fold, fold_df in curves.groupby("fold"):
        axes[0].plot(fold_df["epoch"], fold_df["train_loss"], label=f"fold {int(fold)}")
        axes[1].plot(
            fold_df["epoch"],
            fold_df["val_balanced_accuracy"],
            label=f"fold {int(fold)}",
        )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Balanced Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(f"Training Curves - {report.model_name}")
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig

