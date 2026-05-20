"""EEGMMIDB loading helpers built on top of MNE.

This module focuses on motor imagery first, while still exposing the run
grouping needed to separate baseline, motor execution, and motor imagery runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

TaskType = Literal["baseline", "motor_execution", "motor_imagery"]
SubjectSelection = Literal["all", "debug"] | int | Sequence[int]


RUN_GROUPS: dict[TaskType, tuple[int, ...]] = {
    "baseline": (1, 2),
    "motor_execution": (3, 5, 7, 9, 11, 13),
    "motor_imagery": (4, 6, 8, 10, 12, 14),
}

DEFAULT_TASK_TYPES: tuple[TaskType, ...] = ("motor_imagery",)
MOABB_CLASS_LABELS: tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "feet",
    "hands",
    "rest",
)

_ANNOTATION_CODES = frozenset({"T0", "T1", "T2"})
_FIST_RUNS = frozenset({3, 4, 7, 8, 11, 12})
_HANDS_FEET_RUNS = frozenset({5, 6, 9, 10, 13, 14})
_LABEL_TO_EVENT_CODE: dict[str, int] = {
    "rest": 1,
    "left_hand": 2,
    "right_hand": 3,
    "hands": 4,
    "feet": 5,
}
_TASK_ORDER: tuple[TaskType, ...] = ("baseline", "motor_execution", "motor_imagery")


@dataclass(frozen=True)
class EpochLoadResult:
    """Container for one task type worth of epochs + metadata."""

    task_type: TaskType
    epochs: mne.Epochs
    metadata: pd.DataFrame


@dataclass(frozen=True)
class RunLoadResult:
    """Container for one loaded run."""

    subject_id: int
    run_id: int
    task_type: TaskType
    edf_path: Path
    raw: mne.io.BaseRaw


def list_available_subjects(dataset_root: str | Path) -> list[int]:
    """Return available subject IDs from an EEGMMIDB root folder."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    subject_pattern = re.compile(r"^S(\d{3})$")
    subject_ids: list[int] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            match = subject_pattern.match(child.name)
            if match:
                subject_ids.append(int(match.group(1)))

    if not subject_ids:
        raise FileNotFoundError(
            "No EEGMMIDB subject folders found. Expected names like S001, S002, ..."
        )
    return subject_ids


def resolve_subjects(
    subjects: SubjectSelection,
    dataset_root: str | Path,
    debug_subject_count: int = 2,
) -> list[int]:
    """Resolve subject selection modes: all, single, list, or debug subset."""
    available = list_available_subjects(dataset_root)
    available_set = set(available)

    if subjects == "all":
        return available

    if subjects == "debug":
        if debug_subject_count < 1:
            raise ValueError("debug_subject_count must be >= 1")
        return available[:debug_subject_count]

    if isinstance(subjects, int):
        resolved = [subjects]
    else:
        resolved = sorted(set(int(subject) for subject in subjects))

    missing = [subject for subject in resolved if subject not in available_set]
    if missing:
        raise ValueError(
            f"Requested subjects not found in dataset root {Path(dataset_root)}: {missing}"
        )
    return resolved


def resolve_task_types(task_types: TaskType | Sequence[TaskType]) -> tuple[TaskType, ...]:
    """Normalize one or many task type inputs while preserving canonical order."""
    if isinstance(task_types, str):
        requested = {task_types}
    else:
        requested = set(task_types)

    invalid = sorted(requested.difference(RUN_GROUPS.keys()))
    if invalid:
        raise ValueError(f"Invalid task types: {invalid}. Valid: {list(RUN_GROUPS)}")

    return tuple(task for task in _TASK_ORDER if task in requested)


def runs_for_task_types(task_types: TaskType | Sequence[TaskType]) -> tuple[int, ...]:
    """Get sorted run IDs for one or more task groups."""
    resolved_task_types = resolve_task_types(task_types)
    run_ids = sorted({run for task in resolved_task_types for run in RUN_GROUPS[task]})
    return tuple(run_ids)


def _run_to_task_type(run_id: int) -> TaskType:
    for task_type, run_ids in RUN_GROUPS.items():
        if run_id in run_ids:
            return task_type
    raise ValueError(f"Unknown EEGMMIDB run id: {run_id}")


def _annotation_to_label(run_id: int, annotation: str) -> str:
    if annotation not in _ANNOTATION_CODES:
        return "unknown"
    if annotation == "T0":
        return "rest"
    if run_id in _FIST_RUNS:
        return "left_hand" if annotation == "T1" else "right_hand"
    if run_id in _HANDS_FEET_RUNS:
        return "hands" if annotation == "T1" else "feet"
    # Baseline runs are rest-focused; map any task marker to rest for consistency.
    return "rest"


def _edf_path(dataset_root: Path, subject_id: int, run_id: int) -> Path:
    return dataset_root / f"S{subject_id:03d}" / f"S{subject_id:03d}R{run_id:02d}.edf"


def _events_and_metadata_for_run(
    raw: mne.io.BaseRaw,
    subject_id: int,
    run_id: int,
) -> tuple[list[list[int]], list[dict[str, str | int]]]:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    inverse_event_id = {code: annotation for annotation, code in event_id.items()}

    remapped_events: list[list[int]] = []
    metadata_rows: list[dict[str, str | int]] = []
    task_type = _run_to_task_type(run_id)

    for sample_idx, prev_value, event_code in events:
        annotation = inverse_event_id.get(event_code)
        if annotation not in _ANNOTATION_CODES:
            continue

        label = _annotation_to_label(run_id, annotation)
        remapped_events.append([sample_idx, prev_value, _LABEL_TO_EVENT_CODE[label]])
        metadata_rows.append(
            {
                "subject_id": int(subject_id),
                "run_id": int(run_id),
                "label": label,
                "task_type": task_type,
                "original_annotation": annotation,
            }
        )

    return remapped_events, metadata_rows


def load_eegmmidb_epochs(
    dataset_root: str | Path = "data/files/eegmmidb/1.0.0",
    subjects: SubjectSelection = "all",
    task_types: TaskType | Sequence[TaskType] = DEFAULT_TASK_TYPES,
    tmin: float = 0.0,
    tmax: float = 4.0,
    baseline: tuple[float | None, float | None] | None = None,
    preload: bool = True,
    standardize_channels: bool = True,
    set_montage: bool = True,
    debug_subject_count: int = 2,
) -> dict[TaskType, EpochLoadResult]:
    """Load EEGMMIDB runs as MNE epochs with per-epoch metadata.

    Parameters
    ----------
    dataset_root:
        Folder containing EEGMMIDB subject folders (`S001`, `S002`, ...).
    subjects:
        `"all"`, `"debug"`, single subject int, or a sequence of subject ints.
    task_types:
        One or more of `baseline`, `motor_execution`, `motor_imagery`.
        Default focuses on `motor_imagery`.
    tmin, tmax, baseline:
        Standard MNE epoching parameters.
    preload:
        Passed to EDF reader and Epochs construction.
    standardize_channels:
        If True, applies `mne.datasets.eegbci.standardize(raw)`.
    set_montage:
        If True, applies `standard_1005` montage with `on_missing='ignore'`.
    debug_subject_count:
        Number of subjects used when `subjects='debug'`.

    Returns
    -------
    dict[TaskType, EpochLoadResult]
        A dictionary keyed by task type. Each value contains an `mne.Epochs`
        object and a metadata dataframe with columns:
        `subject_id`, `run_id`, `label`, `task_type`, `original_annotation`.
    """
    runs_by_task = load_eegmmidb_raw_runs(
        dataset_root=dataset_root,
        subjects=subjects,
        task_types=task_types,
        preload=preload,
        standardize_channels=standardize_channels,
        set_montage=set_montage,
        debug_subject_count=debug_subject_count,
    )

    results: dict[TaskType, EpochLoadResult] = {}
    for task_type, loaded_runs in runs_by_task.items():
        per_run_epochs: list[mne.Epochs] = []

        for run_record in loaded_runs:
            remapped_events, metadata_rows = _events_and_metadata_for_run(
                raw=run_record.raw,
                subject_id=run_record.subject_id,
                run_id=run_record.run_id,
            )
            if not remapped_events:
                continue

            metadata_df = pd.DataFrame(metadata_rows)
            events_array = np.asarray(remapped_events, dtype=int)
            present_codes = sorted(set(events_array[:, 2]))
            present_event_id = {
                label: code
                for label, code in _LABEL_TO_EVENT_CODE.items()
                if code in present_codes
            }

            epochs = mne.Epochs(
                raw=run_record.raw,
                events=events_array,
                event_id=present_event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                preload=preload,
                metadata=metadata_df,
                reject_by_annotation=False,
                verbose=False,
            )
            per_run_epochs.append(epochs)

        if not per_run_epochs:
            raise ValueError(
                f"No epochs were created for task '{task_type}'. "
                f"Check dataset_root={Path(dataset_root)}."
            )

        combined = mne.concatenate_epochs(per_run_epochs, add_offset=True, verbose=False)
        results[task_type] = EpochLoadResult(
            task_type=task_type,
            epochs=combined,
            metadata=combined.metadata.copy().reset_index(drop=True),
        )

    return results


def load_eegmmidb_raw_runs(
    dataset_root: str | Path = "data/files/eegmmidb/1.0.0",
    subjects: SubjectSelection = "all",
    task_types: TaskType | Sequence[TaskType] = DEFAULT_TASK_TYPES,
    preload: bool = True,
    standardize_channels: bool = True,
    set_montage: bool = True,
    debug_subject_count: int = 2,
) -> dict[TaskType, list[RunLoadResult]]:
    """Load EEGMMIDB raw runs grouped by task type."""
    root = Path(dataset_root)
    selected_subjects = resolve_subjects(
        subjects=subjects,
        dataset_root=root,
        debug_subject_count=debug_subject_count,
    )
    selected_task_types = resolve_task_types(task_types)

    grouped_runs: dict[TaskType, list[RunLoadResult]] = {
        task_type: [] for task_type in selected_task_types
    }

    for task_type in selected_task_types:
        for subject_id in selected_subjects:
            for run_id in RUN_GROUPS[task_type]:
                edf_path = _edf_path(root, subject_id, run_id)
                if not edf_path.exists():
                    continue

                raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
                if standardize_channels:
                    eegbci.standardize(raw)
                if set_montage:
                    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)

                grouped_runs[task_type].append(
                    RunLoadResult(
                        subject_id=subject_id,
                        run_id=run_id,
                        task_type=task_type,
                        edf_path=edf_path,
                        raw=raw,
                    )
                )

    return grouped_runs
