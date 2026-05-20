"""Leakage-safe split strategies and reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

SplitMode = Literal["subject_dependent_run", "subject_independent_groupkfold", "subject_independent_logo"]


@dataclass(frozen=True)
class SplitIndices:
    """Single train/test split with metadata for auditability."""

    mode: SplitMode
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]
    train_runs: tuple[int, ...]
    test_runs: tuple[int, ...]


@dataclass(frozen=True)
class PerformanceSummary:
    """Aggregated metrics for one evaluation protocol."""

    protocol: str
    fold_count: int
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]


def _validate_metadata(metadata: pd.DataFrame) -> None:
    required = {"subject_id", "run_id"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    if len(metadata) == 0:
        raise ValueError("Metadata is empty.")


def _as_arrays(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    subject_ids = metadata["subject_id"].to_numpy(dtype=np.int64)
    run_ids = metadata["run_id"].to_numpy(dtype=np.int64)
    return subject_ids, run_ids


def assert_subject_dependent_no_run_leakage(metadata: pd.DataFrame, split: SplitIndices) -> None:
    """Ensure train and test runs do not overlap within each subject."""
    _validate_metadata(metadata)
    if split.mode != "subject_dependent_run":
        raise ValueError("Expected a subject-dependent split.")

    subject_ids, run_ids = _as_arrays(metadata)
    for subject in np.unique(subject_ids):
        subj_train_runs = set(run_ids[split.train_idx][subject_ids[split.train_idx] == subject])
        subj_test_runs = set(run_ids[split.test_idx][subject_ids[split.test_idx] == subject])
        if subj_train_runs.intersection(subj_test_runs):
            raise ValueError(
                f"Run leakage detected for subject {int(subject)}: "
                f"{sorted(subj_train_runs.intersection(subj_test_runs))}"
            )


def assert_subject_independent_no_subject_leakage(
    metadata: pd.DataFrame,
    split: SplitIndices,
) -> None:
    """Ensure train and test subjects are disjoint."""
    _validate_metadata(metadata)
    if split.mode not in {"subject_independent_groupkfold", "subject_independent_logo"}:
        raise ValueError("Expected a subject-independent split.")

    train_subjects = set(metadata.iloc[split.train_idx]["subject_id"].astype(int).tolist())
    test_subjects = set(metadata.iloc[split.test_idx]["subject_id"].astype(int).tolist())
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        raise ValueError(f"Subject leakage detected: {sorted(overlap)}")


def make_subject_dependent_run_split(
    metadata: pd.DataFrame,
    test_run_fraction: float = 0.2,
    random_state: int = 42,
    fold: int = 0,
) -> SplitIndices:
    """Train/test split within each subject using run-wise holdout."""
    _validate_metadata(metadata)
    if not (0.0 < test_run_fraction < 1.0):
        raise ValueError("test_run_fraction must be in (0, 1).")

    subject_ids, run_ids = _as_arrays(metadata)
    rng = np.random.default_rng(random_state + fold)

    train_mask = np.zeros(len(metadata), dtype=bool)
    test_mask = np.zeros(len(metadata), dtype=bool)

    for subject in np.unique(subject_ids):
        subj_mask = subject_ids == subject
        subj_runs = np.unique(run_ids[subj_mask])
        if len(subj_runs) < 2:
            raise ValueError(
                f"Subject {int(subject)} has {len(subj_runs)} run(s). "
                "At least 2 runs are required for run-wise dependent splitting."
            )

        shuffled_runs = rng.permutation(subj_runs)
        n_test_runs = int(np.ceil(len(subj_runs) * test_run_fraction))
        n_test_runs = max(1, min(n_test_runs, len(subj_runs) - 1))

        test_runs = set(shuffled_runs[:n_test_runs].tolist())
        train_runs = set(shuffled_runs[n_test_runs:].tolist())

        subj_train_mask = subj_mask & np.isin(run_ids, list(train_runs))
        subj_test_mask = subj_mask & np.isin(run_ids, list(test_runs))
        train_mask |= subj_train_mask
        test_mask |= subj_test_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    split = SplitIndices(
        mode="subject_dependent_run",
        fold=fold,
        train_idx=train_idx,
        test_idx=test_idx,
        train_subjects=tuple(sorted(np.unique(subject_ids[train_idx]).astype(int).tolist())),
        test_subjects=tuple(sorted(np.unique(subject_ids[test_idx]).astype(int).tolist())),
        train_runs=tuple(sorted(np.unique(run_ids[train_idx]).astype(int).tolist())),
        test_runs=tuple(sorted(np.unique(run_ids[test_idx]).astype(int).tolist())),
    )
    assert_subject_dependent_no_run_leakage(metadata, split)
    return split


def make_subject_independent_splits(
    metadata: pd.DataFrame,
    method: Literal["groupkfold", "logo"] = "groupkfold",
    n_splits: int = 5,
) -> list[SplitIndices]:
    """Subject-independent CV grouped by subject ID."""
    _validate_metadata(metadata)
    subject_ids, run_ids = _as_arrays(metadata)
    groups = subject_ids
    dummy_X = np.zeros((len(metadata), 1), dtype=np.float32)

    splits: list[SplitIndices] = []
    if method == "groupkfold":
        unique_subjects = np.unique(groups)
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for GroupKFold.")
        if n_splits > len(unique_subjects):
            raise ValueError(
                f"n_splits={n_splits} exceeds number of subjects ({len(unique_subjects)})."
            )
        splitter = GroupKFold(n_splits=n_splits)
        mode: SplitMode = "subject_independent_groupkfold"
    elif method == "logo":
        splitter = LeaveOneGroupOut()
        mode = "subject_independent_logo"
    else:
        raise ValueError("method must be 'groupkfold' or 'logo'.")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy_X, groups=groups)):
        split = SplitIndices(
            mode=mode,
            fold=fold,
            train_idx=train_idx.astype(np.int64, copy=False),
            test_idx=test_idx.astype(np.int64, copy=False),
            train_subjects=tuple(sorted(np.unique(subject_ids[train_idx]).astype(int).tolist())),
            test_subjects=tuple(sorted(np.unique(subject_ids[test_idx]).astype(int).tolist())),
            train_runs=tuple(sorted(np.unique(run_ids[train_idx]).astype(int).tolist())),
            test_runs=tuple(sorted(np.unique(run_ids[test_idx]).astype(int).tolist())),
        )
        assert_subject_independent_no_subject_leakage(metadata, split)
        splits.append(split)

    return splits


def summarize_fold_metrics(
    protocol: str,
    fold_metrics: Sequence[dict[str, float]],
) -> PerformanceSummary:
    """Aggregate fold metrics into mean/std summary."""
    if len(fold_metrics) == 0:
        raise ValueError("fold_metrics must not be empty.")

    metric_names = sorted(set().union(*(metrics.keys() for metrics in fold_metrics)))
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for name in metric_names:
        values = np.array([float(metrics[name]) for metrics in fold_metrics if name in metrics], dtype=float)
        mean_metrics[name] = float(values.mean())
        std_metrics[name] = float(values.std(ddof=0))

    return PerformanceSummary(
        protocol=protocol,
        fold_count=len(fold_metrics),
        metrics_mean=mean_metrics,
        metrics_std=std_metrics,
    )


def make_generalization_report(
    subject_dependent_summary: PerformanceSummary,
    subject_independent_summary: PerformanceSummary,
) -> pd.DataFrame:
    """Return a compact table reporting both dependent and independent results."""
    all_metrics = sorted(
        set(subject_dependent_summary.metrics_mean).union(subject_independent_summary.metrics_mean)
    )

    rows: list[dict[str, float | str | int]] = []
    for summary in (subject_dependent_summary, subject_independent_summary):
        row: dict[str, float | str | int] = {
            "protocol": summary.protocol,
            "fold_count": summary.fold_count,
        }
        for metric in all_metrics:
            row[f"{metric}_mean"] = summary.metrics_mean.get(metric, np.nan)
            row[f"{metric}_std"] = summary.metrics_std.get(metric, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)

