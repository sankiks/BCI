"""Prediction task definitions for EEG motor imagery benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import mne
import numpy as np
import pandas as pd

from src.preprocessing import EpochLoadResult

TaskKey = Literal["task_a_binary_lr", "task_b_four_class_mi", "task_c_five_class_with_rest"]


@dataclass(frozen=True)
class PredictionTaskSpec:
    """Definition of one prediction task."""

    key: TaskKey
    name: str
    class_labels: tuple[str, ...]
    requires_balancing: bool
    optional: bool


@dataclass(frozen=True)
class PreparedTaskDataset:
    """Task-specific view over epochs + labels + balancing utilities."""

    spec: PredictionTaskSpec
    epochs: mne.Epochs
    metadata: pd.DataFrame
    y: np.ndarray
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]
    class_counts: dict[str, int]
    class_weights: dict[int, float] | None
    sample_weights: np.ndarray | None


TASK_A_BINARY_LR = PredictionTaskSpec(
    key="task_a_binary_lr",
    name="Task A: left hand vs right hand motor imagery",
    class_labels=("left_hand", "right_hand"),
    requires_balancing=False,
    optional=False,
)

TASK_B_FOUR_CLASS_MI = PredictionTaskSpec(
    key="task_b_four_class_mi",
    name="Task B: 4-class motor imagery",
    class_labels=("left_hand", "right_hand", "hands", "feet"),
    requires_balancing=False,
    optional=False,
)

TASK_C_FIVE_CLASS_WITH_REST = PredictionTaskSpec(
    key="task_c_five_class_with_rest",
    name="Task C: 5-class including rest",
    class_labels=("left_hand", "right_hand", "hands", "feet", "rest"),
    requires_balancing=True,
    optional=True,
)

TASK_SPECS_IN_ORDER: tuple[PredictionTaskSpec, ...] = (
    TASK_A_BINARY_LR,
    TASK_B_FOUR_CLASS_MI,
    TASK_C_FIVE_CLASS_WITH_REST,
)
TASK_SPEC_BY_KEY: dict[TaskKey, PredictionTaskSpec] = {spec.key: spec for spec in TASK_SPECS_IN_ORDER}


def _balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    n_classes = float(len(classes))
    # Standard "balanced" weighting used to offset class-frequency imbalance.
    return {int(cls): total / (n_classes * float(count)) for cls, count in zip(classes, counts)}


def balanced_sample_indices(y: np.ndarray, random_state: int | None = None) -> np.ndarray:
    """Return undersampled indices with equal sample count per class."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target_count = int(np.min(counts))

    selected_parts: list[np.ndarray] = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        picked = rng.choice(cls_indices, size=target_count, replace=False)
        selected_parts.append(picked)

    indices = np.concatenate(selected_parts)
    rng.shuffle(indices)
    return indices


def prepare_prediction_task(
    epoch_result: EpochLoadResult,
    task: TaskKey | PredictionTaskSpec,
    include_sample_weights: bool = True,
) -> PreparedTaskDataset:
    """Prepare one task-specific dataset from motor-imagery epochs."""
    spec = task if isinstance(task, PredictionTaskSpec) else TASK_SPEC_BY_KEY[task]
    metadata = epoch_result.metadata.reset_index(drop=True).copy()

    if "label" not in metadata.columns:
        raise ValueError("Epoch metadata must contain a 'label' column.")

    keep_mask = metadata["label"].isin(spec.class_labels)
    keep_indices = np.where(keep_mask.to_numpy())[0]
    if len(keep_indices) == 0:
        raise ValueError(f"No samples found for {spec.name}.")

    task_epochs = epoch_result.epochs[keep_indices]
    task_metadata = task_epochs.metadata.copy().reset_index(drop=True)

    label_to_index = {label: idx for idx, label in enumerate(spec.class_labels)}
    y = task_metadata["label"].map(label_to_index).to_numpy(dtype=np.int64)

    class_counts = {
        label: int((task_metadata["label"] == label).sum())
        for label in spec.class_labels
    }

    class_weights: dict[int, float] | None = None
    sample_weights: np.ndarray | None = None
    if spec.requires_balancing:
        class_weights = _balanced_class_weights(y)
        if include_sample_weights:
            sample_weights = np.array([class_weights[int(target)] for target in y], dtype=np.float64)

    return PreparedTaskDataset(
        spec=spec,
        epochs=task_epochs,
        metadata=task_metadata,
        y=y,
        label_to_index=label_to_index,
        index_to_label={idx: label for label, idx in label_to_index.items()},
        class_counts=class_counts,
        class_weights=class_weights,
        sample_weights=sample_weights,
    )


def prepare_prediction_tasks(
    epoch_result: EpochLoadResult,
    include_optional: bool = True,
    include_sample_weights: bool = True,
) -> list[PreparedTaskDataset]:
    """Prepare task datasets in required order: A, B, then C."""
    prepared: list[PreparedTaskDataset] = []
    for spec in TASK_SPECS_IN_ORDER:
        if spec.optional and not include_optional:
            continue
        prepared.append(
            prepare_prediction_task(
                epoch_result=epoch_result,
                task=spec,
                include_sample_weights=include_sample_weights,
            )
        )
    return prepared

