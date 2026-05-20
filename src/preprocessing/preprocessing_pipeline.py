"""EEG preprocessing pipeline for EEGMMIDB motor-imagery experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import mne
import numpy as np
import pandas as pd

from .eegmmidb_loader import (
    EpochLoadResult,
    RunLoadResult,
    SubjectSelection,
    TaskType,
    load_eegmmidb_raw_runs,
)

BandpassMode = Literal["default", "wide"]

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
_BANDPASS_PRESETS: dict[BandpassMode, tuple[float, float]] = {
    "default": (8.0, 30.0),
    "wide": (4.0, 40.0),
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Signal and epoching configuration for preprocessing."""

    target_sfreq: float = 160.0
    bandpass: tuple[float, float] = (8.0, 30.0)
    notch_freqs: tuple[float, ...] | None = None
    epoch_tmin: float = 0.5
    epoch_tmax: float = 3.5
    include_tmax_endpoint: bool = False
    baseline: tuple[float | None, float | None] | None = None
    expected_n_channels: int = 64
    expected_n_times: int = 480
    resample_tolerance_hz: float = 1e-3


@dataclass(frozen=True)
class NormalizationStats:
    """Per-subject per-channel normalization stats built from training data only."""

    subject_means: dict[int, np.ndarray]
    subject_stds: dict[int, np.ndarray]
    global_mean: np.ndarray
    global_std: np.ndarray
    train_indices: np.ndarray


@dataclass(frozen=True)
class PreprocessedDataset:
    """Preprocessed arrays ready for model training and evaluation."""

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    metadata: pd.DataFrame
    label_to_index: dict[str, int]
    normalization: NormalizationStats
    output_dir: Path | None = None


def make_preprocessing_config(
    bandpass_mode: BandpassMode = "default",
    notch_freqs: Sequence[float] | None = None,
    target_sfreq: float = 160.0,
    epoch_tmin: float = 0.5,
    epoch_tmax: float = 3.5,
    include_tmax_endpoint: bool = False,
    expected_n_channels: int = 64,
    expected_n_times: int = 480,
) -> PreprocessingConfig:
    """Create a config using named bandpass presets (`default`, `wide`)."""
    if bandpass_mode not in _BANDPASS_PRESETS:
        raise ValueError(f"Unknown bandpass_mode='{bandpass_mode}'. Valid: {list(_BANDPASS_PRESETS)}")
    notch = None if notch_freqs is None else tuple(float(freq) for freq in notch_freqs)
    return PreprocessingConfig(
        target_sfreq=float(target_sfreq),
        bandpass=_BANDPASS_PRESETS[bandpass_mode],
        notch_freqs=notch,
        epoch_tmin=float(epoch_tmin),
        epoch_tmax=float(epoch_tmax),
        include_tmax_endpoint=bool(include_tmax_endpoint),
        expected_n_channels=int(expected_n_channels),
        expected_n_times=int(expected_n_times),
    )


def _annotation_to_label(run_id: int, annotation: str) -> str:
    if annotation not in _ANNOTATION_CODES:
        return "unknown"
    if annotation == "T0":
        return "rest"
    if run_id in _FIST_RUNS:
        return "left_hand" if annotation == "T1" else "right_hand"
    if run_id in _HANDS_FEET_RUNS:
        return "hands" if annotation == "T1" else "feet"
    return "rest"


def _preprocess_raw_signal(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> mne.io.BaseRaw:
    processed = raw.copy()
    current_sfreq = float(processed.info["sfreq"])
    if abs(current_sfreq - config.target_sfreq) > config.resample_tolerance_hz:
        processed.resample(config.target_sfreq, npad="auto", verbose=False)

    if config.notch_freqs:
        nyquist = 0.5 * float(processed.info["sfreq"])
        valid_notch = [freq for freq in config.notch_freqs if 0.0 < freq < nyquist]
        if valid_notch:
            processed.notch_filter(freqs=np.array(valid_notch, dtype=float), picks="eeg", verbose=False)

    l_freq, h_freq = config.bandpass
    processed.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
    return processed


def _effective_epoch_tmax(config: PreprocessingConfig) -> float:
    if config.include_tmax_endpoint:
        return config.epoch_tmax
    # MNE includes end samples by default; subtract one sample to get exact window length.
    return config.epoch_tmax - (1.0 / config.target_sfreq)


def _epoch_single_run(run: RunLoadResult, config: PreprocessingConfig) -> mne.Epochs | None:
    raw = _preprocess_raw_signal(run.raw, config)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    inverse_event_id = {code: annotation for annotation, code in event_id.items()}

    remapped_events: list[list[int]] = []
    metadata_rows: list[dict[str, str | int]] = []
    for sample_idx, prev_value, event_code in events:
        annotation = inverse_event_id.get(event_code)
        if annotation not in _ANNOTATION_CODES:
            continue

        label = _annotation_to_label(run.run_id, annotation)
        remapped_events.append([sample_idx, prev_value, _LABEL_TO_EVENT_CODE[label]])
        metadata_rows.append(
            {
                "subject_id": int(run.subject_id),
                "run_id": int(run.run_id),
                "label": label,
                "task_type": run.task_type,
                "original_annotation": annotation,
            }
        )

    if not remapped_events:
        return None

    metadata_df = pd.DataFrame(metadata_rows)
    events_array = np.asarray(remapped_events, dtype=int)
    present_codes = sorted(set(events_array[:, 2]))
    present_event_id = {
        label: code
        for label, code in _LABEL_TO_EVENT_CODE.items()
        if code in present_codes
    }

    epochs = mne.Epochs(
        raw=raw,
        events=events_array,
        event_id=present_event_id,
        tmin=config.epoch_tmin,
        tmax=_effective_epoch_tmax(config),
        baseline=config.baseline,
        preload=True,
        metadata=metadata_df,
        reject_by_annotation=False,
        verbose=False,
    )
    return epochs


def create_preprocessed_epochs(
    dataset_root: str | Path = "data/files/eegmmidb/1.0.0",
    subjects: SubjectSelection = "all",
    task_type: TaskType = "motor_imagery",
    config: PreprocessingConfig | None = None,
    standardize_channels: bool = True,
    set_montage: bool = True,
    debug_subject_count: int = 2,
) -> EpochLoadResult:
    """Load, filter, resample (if needed), and epoch EEGMMIDB runs."""
    active_config = config or PreprocessingConfig()
    runs_by_task = load_eegmmidb_raw_runs(
        dataset_root=dataset_root,
        subjects=subjects,
        task_types=(task_type,),
        preload=True,
        standardize_channels=standardize_channels,
        set_montage=set_montage,
        debug_subject_count=debug_subject_count,
    )
    runs = runs_by_task[task_type]
    if not runs:
        raise ValueError("No runs available after loading raw data.")

    epoch_list: list[mne.Epochs] = []
    for run in runs:
        run_epochs = _epoch_single_run(run, active_config)
        if run_epochs is not None and len(run_epochs) > 0:
            epoch_list.append(run_epochs)

    if not epoch_list:
        raise ValueError("No epochs created from loaded runs.")

    combined = mne.concatenate_epochs(epoch_list, add_offset=True, verbose=False)
    n_channels = len(combined.ch_names)
    n_times = len(combined.times)
    if n_channels != active_config.expected_n_channels or n_times != active_config.expected_n_times:
        raise ValueError(
            "Unexpected epoch shape. "
            f"Expected ({active_config.expected_n_channels}, {active_config.expected_n_times}), "
            f"got ({n_channels}, {n_times})."
        )

    return EpochLoadResult(
        task_type=task_type,
        epochs=combined,
        metadata=combined.metadata.copy().reset_index(drop=True),
    )


def _resolve_train_indices(n_samples: int, train_indices: Sequence[int] | None) -> np.ndarray:
    if train_indices is None:
        return np.arange(n_samples, dtype=np.int64)
    idx = np.asarray(train_indices, dtype=np.int64)
    if idx.ndim != 1 or len(idx) == 0:
        raise ValueError("train_indices must be a non-empty 1D sequence.")
    if np.min(idx) < 0 or np.max(idx) >= n_samples:
        raise ValueError("train_indices contains out-of-range indices.")
    return np.unique(idx)


def compute_subject_channel_stats(
    X: np.ndarray,
    subjects: np.ndarray,
    train_indices: Sequence[int] | None = None,
    eps: float = 1e-6,
) -> NormalizationStats:
    """Compute per-subject per-channel mean/std from training data only."""
    if X.ndim != 3:
        raise ValueError("X must be 3D: (n_epochs, n_channels, n_times).")
    n_samples = X.shape[0]
    if len(subjects) != n_samples:
        raise ValueError("subjects length must match X.shape[0].")

    train_idx = _resolve_train_indices(n_samples, train_indices)
    train_X = X[train_idx]
    train_subjects = subjects[train_idx]

    global_mean = train_X.mean(axis=(0, 2))
    global_std = train_X.std(axis=(0, 2))
    global_std = np.where(global_std < eps, 1.0, global_std)

    subject_means: dict[int, np.ndarray] = {}
    subject_stds: dict[int, np.ndarray] = {}
    for subject in np.unique(train_subjects):
        subj_mask = train_subjects == subject
        subj_X = train_X[subj_mask]
        subj_mean = subj_X.mean(axis=(0, 2))
        subj_std = subj_X.std(axis=(0, 2))
        subj_std = np.where(subj_std < eps, 1.0, subj_std)
        subject_means[int(subject)] = subj_mean
        subject_stds[int(subject)] = subj_std

    return NormalizationStats(
        subject_means=subject_means,
        subject_stds=subject_stds,
        global_mean=global_mean,
        global_std=global_std,
        train_indices=train_idx,
    )


def apply_subject_channel_zscore(
    X: np.ndarray,
    subjects: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """Apply z-score normalization per subject and channel."""
    X_norm = np.empty_like(X, dtype=np.float32)
    for idx, subject in enumerate(subjects):
        subj_key = int(subject)
        mean = stats.subject_means.get(subj_key, stats.global_mean)
        std = stats.subject_stds.get(subj_key, stats.global_std)
        X_norm[idx] = ((X[idx] - mean[:, None]) / std[:, None]).astype(np.float32)
    return X_norm


def prepare_and_save_arrays(
    epoch_result: EpochLoadResult,
    output_dir: str | Path,
    class_labels: Sequence[str],
    train_indices: Sequence[int] | None = None,
) -> PreprocessedDataset:
    """Convert epochs to arrays, normalize, and save required files."""
    metadata = epoch_result.metadata.reset_index(drop=True).copy()
    if "label" not in metadata.columns:
        raise ValueError("Epoch metadata must include 'label'.")
    if "subject_id" not in metadata.columns or "run_id" not in metadata.columns:
        raise ValueError("Epoch metadata must include 'subject_id' and 'run_id'.")

    ordered_labels = tuple(class_labels)
    keep_mask = metadata["label"].isin(ordered_labels)
    keep_indices = np.where(keep_mask.to_numpy())[0]
    if len(keep_indices) == 0:
        raise ValueError("No epochs matched the requested class labels.")

    epochs = epoch_result.epochs[keep_indices]
    selected_metadata = epochs.metadata.copy().reset_index(drop=True)
    label_to_index = {label: idx for idx, label in enumerate(ordered_labels)}
    y = selected_metadata["label"].map(label_to_index).to_numpy(dtype=np.int64)
    selected_metadata["target_index"] = y

    X = epochs.get_data(copy=True).astype(np.float32)
    subjects = selected_metadata["subject_id"].to_numpy(dtype=np.int64)
    runs = selected_metadata["run_id"].to_numpy(dtype=np.int64)

    stats = compute_subject_channel_stats(X=X, subjects=subjects, train_indices=train_indices)
    X_norm = apply_subject_channel_zscore(X=X, subjects=subjects, stats=stats)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X_norm)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subjects.npy", subjects)
    np.save(out_dir / "runs.npy", runs)
    selected_metadata.to_csv(out_dir / "metadata.csv", index=False)

    return PreprocessedDataset(
        X=X_norm,
        y=y,
        subjects=subjects,
        runs=runs,
        metadata=selected_metadata,
        label_to_index=label_to_index,
        normalization=stats,
        output_dir=out_dir,
    )
