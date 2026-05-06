from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mne.datasets import eegbci
from torch.utils.data import DataLoader, Dataset


def extract_run_number(edf_path: Path) -> int:
    stem = edf_path.stem.upper()
    if "R" in stem:
        run_part = stem.rsplit("R", 1)[-1]
        if run_part.isdigit() and len(run_part) == 2:
            return int(run_part)
    m = re.search(r"R([0-9]{2})", stem)
    if m is None:
        raise ValueError(f"Could not parse run id from: {edf_path.name}")
    return int(m.group(1))


def subject_id_from_edf_path(edf_path: Path) -> int:
    m = re.search(r"S([0-9]{3})", edf_path.parent.name.upper())
    if m is not None:
        return int(m.group(1))
    m = re.search(r"S([0-9]{3})", edf_path.stem.upper())
    if m is None:
        raise ValueError(f"Could not parse subject id from: {edf_path}")
    return int(m.group(1))


def discover_edf_files(
    data_root: Path,
    subject_ids: List[int],
    runs: List[int],
    max_files: int | None = None,
) -> List[Path]:
    run_set = set(runs)
    files: List[Path] = []
    for sid in subject_ids:
        subject_dir = data_root / f"S{sid:03d}"
        if not subject_dir.exists():
            continue
        for edf_path in sorted(subject_dir.glob("*.edf")):
            if extract_run_number(edf_path) in run_set:
                files.append(edf_path)
    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]
    return files


def preprocess_raw(
    raw: mne.io.BaseRaw,
    event_rename_map: Dict[str, str],
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    target_sfreq: float = 128.0,
) -> mne.io.BaseRaw:
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="warn")
    raw.set_eeg_reference(projection=True)
    if target_sfreq is not None and abs(raw.info["sfreq"] - target_sfreq) > 1e-6:
        raw.resample(target_sfreq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", skip_by_annotation="edge")
    raw.annotations.rename(event_rename_map)
    return raw


def epochs_from_raw(
    raw: mne.io.BaseRaw,
    event_map: Dict[str, str],
    tmin: float = 0.0,
    tmax: float = 4.0,
):
    events, full_event_id = mne.events_from_annotations(raw, verbose=False)
    wanted_labels = list(event_map.values())
    event_id = {label: full_event_id[label] for label in wanted_labels if label in full_event_id}
    if not event_id:
        return None, None
    keep_ids = set(event_id.values())
    events = events[np.isin(events[:, -1], list(keep_ids))]
    if len(events) == 0:
        return None, None
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        proj=True,
        picks=picks,
        verbose=False,
    )
    return epochs, event_id


def build_epoch_arrays(
    edf_files: List[Path],
    event_map: Dict[str, str],
    l_freq: float,
    h_freq: float,
    tmin: float,
    tmax: float,
    target_sfreq: float = 128.0,
):
    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    subj_chunks: List[np.ndarray] = []
    run_chunks: List[np.ndarray] = []
    observed_event_id = None

    for i, edf_path in enumerate(edf_files, start=1):
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw = preprocess_raw(raw, event_map, l_freq=l_freq, h_freq=h_freq, target_sfreq=target_sfreq)
        epochs, event_id = epochs_from_raw(raw, event_map, tmin=tmin, tmax=tmax)
        if epochs is None:
            continue
        if observed_event_id is None:
            observed_event_id = event_id

        X = epochs.get_data(copy=False).astype(np.float32)
        y = epochs.events[:, -1].astype(np.int64)
        sid = subject_id_from_edf_path(edf_path)
        run_no = extract_run_number(edf_path)

        X_chunks.append(X)
        y_chunks.append(y)
        subj_chunks.append(np.full(shape=(len(y),), fill_value=sid, dtype=np.int32))
        run_chunks.append(np.full(shape=(len(y),), fill_value=run_no, dtype=np.int16))

        if i % 20 == 0:
            print(f"Processed {i}/{len(edf_files)} files")

    if not X_chunks:
        raise RuntimeError("No epochs were extracted. Check task/runs/data path.")

    X_all = np.concatenate(X_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    subject_ids = np.concatenate(subj_chunks, axis=0)
    run_ids = np.concatenate(run_chunks, axis=0)

    unique_ids = np.unique(y_all)
    id_to_class = {eid: idx for idx, eid in enumerate(unique_ids)}
    y_all = np.vectorize(id_to_class.get)(y_all).astype(np.int64)
    X_all = X_all[:, np.newaxis, :, :]

    class_to_label: Dict[int, str] = {}
    if observed_event_id is not None:
        label_by_event_id = {eid: label for label, eid in observed_event_id.items()}
        class_to_label = {id_to_class[eid]: label_by_event_id.get(eid, str(eid)) for eid in unique_ids}

    return X_all, y_all, subject_ids, run_ids, id_to_class, class_to_label


def stratified_epoch_split_indices(y: np.ndarray, val_fraction: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def run_holdout_split_indices(run_ids: np.ndarray, holdout_run: int):
    val_idx = np.where(run_ids == holdout_run)[0]
    train_idx = np.where(run_ids != holdout_run)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"Run-holdout split failed for holdout_run={holdout_run}. "
            f"Train={len(train_idx)} | Eval={len(val_idx)}"
        )
    return train_idx, val_idx


def fit_channel_zscore(X_train: np.ndarray, eps: float = 1e-6):
    mean = X_train.mean(axis=(0, 1, 3), keepdims=True)
    std = X_train.std(axis=(0, 1, 3), keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


def sliding_window_augment(
    X: np.ndarray,
    y: np.ndarray,
    window_size_samples: int,
    step_samples: int,
):
    n_trials, _, _, n_times = X.shape
    if window_size_samples >= n_times:
        return X, y
    starts = np.arange(0, n_times - window_size_samples + 1, step_samples, dtype=np.int32)
    if len(starts) <= 1:
        return X, y

    X_out = []
    y_out = []
    for i in range(n_trials):
        for s in starts:
            e = s + window_size_samples
            X_out.append(X[i : i + 1, :, :, s:e])
            y_out.append(y[i])
    return np.concatenate(X_out, axis=0).astype(np.float32), np.asarray(y_out, dtype=np.int64)


class EEGTorchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int,
    num_workers: int,
):
    train_ds = EEGTorchDataset(X_train, y_train)
    eval_ds = EEGTorchDataset(X_eval, y_eval)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, eval_ds, train_loader, eval_loader


class EEGNetLite(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        f1: int = 16,
        d: int = 2,
        f2: int = 32,
        kernel_length: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            feat_dim = self.block2(self.block1(dummy)).reshape(1, -1).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def balanced_accuracy_from_cm(cm: np.ndarray):
    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i, :].sum()
        recalls.append(0.0 if denom == 0 else cm[i, i] / denom)
    return float(np.mean(recalls))


def evaluate(model, loader, criterion, device, n_classes: int):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * yb.size(0)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            all_true.append(yb.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion_matrix_np(y_true, y_pred, n_classes=n_classes)
    return total_loss / total, correct / total, balanced_accuracy_from_cm(cm), y_true, y_pred, cm


def maybe_load_pretrained_backbone(model: nn.Module, ckpt_path: str):
    p = Path(ckpt_path)
    if not p.exists():
        print(f"Pretrain checkpoint not found: {p}. Training from scratch.")
        return

    state = torch.load(p, map_location="cpu")
    model_state = model.state_dict()
    compatible = {
        k: v
        for k, v in state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(
        f"Loaded pretrained weights from {p.name}: "
        f"{len(compatible)} tensors matched, {len(model_state) - len(compatible)} left random."
    )


def train_one_split(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    protocol_name: str,
    *,
    batch_size: int,
    num_workers: int,
    model_size: str = "small",
    use_pretrain: bool = False,
    pretrain_ckpt: str = "best_eegnet_lite_subject_independent.pt",
    class_weight_mode: str = "none",
    class_weights: np.ndarray | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 2e-4,
    grad_clip_norm: float | None = None,
    seed: int | None = None,
    deterministic: bool = True,
    epochs: int = 30,
    early_stopping_patience: int = 20,
    use_early_stopping: bool = False,
    save_best: bool = True,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = X_train_np.shape[2]
    train_times = int(X_train_np.shape[3])
    eval_times = int(X_eval_np.shape[3])
    n_times = train_times
    if eval_times != train_times:
        common_t = min(train_times, eval_times)
        # Final safety alignment for any upstream split/windowing mismatch.
        if train_times > common_t:
            start = (train_times - common_t) // 2
            X_train_np = X_train_np[:, :, :, start : start + common_t]
        if eval_times > common_t:
            start = (eval_times - common_t) // 2
            X_eval_np = X_eval_np[:, :, :, start : start + common_t]
        n_times = common_t
        print(
            f"[{protocol_name}] Auto-aligned temporal length to T={common_t} "
            f"(train was {train_times}, eval was {eval_times})."
        )
    n_classes = int(max(y_train_np.max(), y_eval_np.max()) + 1)

    model_cfg_by_size = {
        "small": dict(f1=8, d=2, f2=16, kernel_length=64, dropout=0.5),
        "base": dict(f1=16, d=2, f2=32, kernel_length=64, dropout=0.4),
    }
    if model_size not in model_cfg_by_size:
        raise ValueError(f"Unknown model_size: {model_size}")

    model = EEGNetLite(n_channels=n_channels, n_times=n_times, n_classes=n_classes, **model_cfg_by_size[model_size]).to(
        device
    )
    if use_pretrain:
        maybe_load_pretrained_backbone(model, pretrain_ckpt)

    train_ds, eval_ds, train_loader_local, eval_loader_local = make_dataloaders(
        X_train_np, y_train_np, X_eval_np, y_eval_np, batch_size, num_workers
    )

    valid_weight_modes = {"none", "balanced", "custom"}
    if class_weight_mode not in valid_weight_modes:
        raise ValueError(f"class_weight_mode must be one of {sorted(valid_weight_modes)}, got {class_weight_mode}")

    if class_weights is None and class_weight_mode == "balanced":
        counts = np.bincount(y_train_np.astype(np.int64), minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weights = (counts.sum() / (n_classes * counts)).astype(np.float32)
    elif class_weights is None and class_weight_mode == "custom":
        raise ValueError("class_weight_mode='custom' requires explicit class_weights.")
    elif class_weights is not None:
        class_weights = np.asarray(class_weights, dtype=np.float32)
        if class_weights.shape[0] != n_classes:
            raise ValueError(f"class_weights length {class_weights.shape[0]} != n_classes {n_classes}")

    weight_tensor = None
    if class_weights is not None and class_weight_mode != "none":
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"[{protocol_name}] class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5)

    history = {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": [], "eval_bal_acc": [], "lr": []}
    best_bal_acc = -np.inf
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        for xb, yb in train_loader_local:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * yb.size(0)
            running_total += yb.size(0)
            running_correct += (preds == yb).sum().item()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        eval_loss, eval_acc, eval_bal_acc, _, _, _ = evaluate(
            model, eval_loader_local, criterion, device, n_classes=n_classes
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)
        history["eval_bal_acc"].append(eval_bal_acc)
        history["lr"].append(current_lr)

        improved = eval_bal_acc > best_bal_acc + 1e-6
        if improved:
            best_bal_acc = eval_bal_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step(eval_bal_acc)

        print(
            f"[{protocol_name}] Epoch {epoch:02d}/{epochs} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} eval_bal_acc={eval_bal_acc:.4f}"
        )
        if use_early_stopping and no_improve >= early_stopping_patience:
            print(f"[{protocol_name}] Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    final_eval_loss, final_eval_acc, final_eval_bal_acc, _, _, cm = evaluate(
        model, eval_loader_local, criterion, device, n_classes=n_classes
    )

    ckpt_path = f"best_eegnet_lite_{protocol_name}.pt"
    if save_best:
        torch.save(best_state, ckpt_path)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "final_eval_loss": final_eval_loss,
        "final_eval_acc": final_eval_acc,
        "final_eval_bal_acc": final_eval_bal_acc,
        "cm": cm,
        "ckpt_path": ckpt_path if save_best else None,
        "train_count": len(train_ds),
        "eval_count": len(eval_ds),
    }


def prepare_holdout_inputs(
    X_subj: np.ndarray,
    y_subj: np.ndarray,
    runs_subj: np.ndarray,
    holdout_run: int,
    *,
    normalize_channels: bool = True,
    use_augmentation: bool = False,
    augment_eval: bool = False,
    aug_window_seconds: float = 2.0,
    aug_step_seconds: float = 0.5,
    target_sfreq: float = 128.0,
):
    train_idx, eval_idx = run_holdout_split_indices(runs_subj, holdout_run=holdout_run)
    X_train = X_subj[train_idx].copy()
    y_train = y_subj[train_idx].copy()
    X_eval = X_subj[eval_idx].copy()
    y_eval = y_subj[eval_idx].copy()

    # Some run splits can differ by 1 sample after preprocessing/resampling.
    # Align both sets to a shared temporal length to avoid shape mismatches later.
    train_t = int(X_train.shape[-1])
    eval_t = int(X_eval.shape[-1])
    if train_t != eval_t:
        common_t = min(train_t, eval_t)
        if train_t > common_t:
            start = (train_t - common_t) // 2
            X_train = X_train[:, :, :, start : start + common_t]
        if eval_t > common_t:
            start = (eval_t - common_t) // 2
            X_eval = X_eval[:, :, :, start : start + common_t]

    if normalize_channels:
        mean, std = fit_channel_zscore(X_train)
        X_train = apply_channel_zscore(X_train, mean, std)
        X_eval = apply_channel_zscore(X_eval, mean, std)

    if use_augmentation:
        window = int(round(aug_window_seconds * target_sfreq))
        step = int(round(aug_step_seconds * target_sfreq))
        X_train, y_train = sliding_window_augment(
            X_train, y_train, window_size_samples=window, step_samples=step
        )
        if augment_eval:
            X_eval, y_eval = sliding_window_augment(
                X_eval, y_eval, window_size_samples=window, step_samples=step
            )
        else:
            # Keep holdout evaluation non-augmented but shape-compatible with the model.
            # If train windows are shorter than full trials, center-crop eval to same T.
            eval_t = X_eval.shape[-1]
            if window < eval_t:
                start = (eval_t - window) // 2
                end = start + window
                X_eval = X_eval[:, :, :, start:end]

    return X_train, y_train, X_eval, y_eval


def _merge_cfg(base: Dict[str, Any], override: Dict[str, Any] | None):
    out = dict(base)
    if override:
        out.update(override)
    return out


def run_adaptive_holdout_sweep(
    X_subj: np.ndarray,
    y_subj: np.ndarray,
    runs_subj: np.ndarray,
    holdout_runs: List[int],
    candidate_configs: List[Dict[str, Any]],
    *,
    train_kwargs: Dict[str, Any],
    data_kwargs: Dict[str, Any] | None = None,
    initial_epochs: int = 12,
    max_epochs: int = 80,
    max_stages: int = 3,
    keep_ratio: float = 0.5,
    min_keep: int = 2,
    growth_factor: float = 1.8,
    min_stage_improvement: float = 0.003,
    early_stopping_patience: int = 6,
    patience_growth: float = 1.4,
    use_early_stopping: bool = True,
    deterministic: bool = True,
    base_seed: int = 42,
    seeds_per_candidate: int = 1,
):
    if len(candidate_configs) == 0:
        raise ValueError("candidate_configs must contain at least one config.")
    if len(holdout_runs) == 0:
        raise ValueError("holdout_runs must contain at least one run id.")
    if initial_epochs < 1 or max_epochs < initial_epochs:
        raise ValueError("Require 1 <= initial_epochs <= max_epochs.")
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")
    if seeds_per_candidate < 1:
        raise ValueError("seeds_per_candidate must be >= 1.")

    data_kwargs = {} if data_kwargs is None else dict(data_kwargs)
    tracked = []
    for idx, cfg in enumerate(candidate_configs):
        if "name" not in cfg:
            raise ValueError(f"Config at index {idx} is missing required key: 'name'")
        tracked.append(
            {
                "name": str(cfg["name"]),
                "train_overrides": dict(cfg.get("train_overrides", {})),
                "data_overrides": dict(cfg.get("data_overrides", {})),
                "epochs": int(cfg.get("initial_epochs", initial_epochs)),
                "prev_score": -np.inf,
                "stages": [],
            }
        )

    all_stage_rows: List[Dict[str, Any]] = []
    active = tracked

    for stage_idx in range(1, max_stages + 1):
        if len(active) == 0:
            break

        stage_patience = max(2, int(round(early_stopping_patience * (patience_growth ** (stage_idx - 1)))))
        stage_rows = []
        print(f"\n[adaptive] Stage {stage_idx}/{max_stages} | active={len(active)}")

        for cand_i, cand in enumerate(active, start=1):
            cfg_data = _merge_cfg(data_kwargs, cand["data_overrides"])
            cfg_train = _merge_cfg(train_kwargs, cand["train_overrides"])
            epochs_local = min(int(cand["epochs"]), max_epochs)

            run_metrics = []
            run_results = []
            cm_sum = None

            for run_j, holdout_run in enumerate(holdout_runs, start=1):
                run_seed_metrics = []
                for seed_k in range(seeds_per_candidate):
                    Xtr, ytr, Xte, yte = prepare_holdout_inputs(
                        X_subj,
                        y_subj,
                        runs_subj,
                        holdout_run,
                        **cfg_data,
                    )
                    out = train_one_split(
                        Xtr,
                        ytr,
                        Xte,
                        yte,
                        protocol_name=f"adaptive_{cand['name']}_S{stage_idx}_R{holdout_run:02d}_K{seed_k+1}",
                        epochs=epochs_local,
                        early_stopping_patience=stage_patience,
                        use_early_stopping=use_early_stopping,
                        deterministic=deterministic,
                        seed=base_seed + 10000 * stage_idx + 1000 * cand_i + 100 * run_j + seed_k,
                        save_best=False,
                        **cfg_train,
                    )
                    run_seed_metrics.append(float(out["final_eval_bal_acc"]))
                    run_results.append(out)
                    cm_sum = out["cm"] if cm_sum is None else (cm_sum + out["cm"])
                run_metrics.append(float(np.mean(run_seed_metrics)))

            mean_bal = float(np.mean(run_metrics))
            std_bal = float(np.std(run_metrics))
            min_bal = float(np.min(run_metrics))
            improvement = mean_bal - float(cand["prev_score"])

            row = {
                "stage": stage_idx,
                "name": cand["name"],
                "epochs": epochs_local,
                "patience": stage_patience,
                "bal_mean": mean_bal,
                "bal_std": std_bal,
                "bal_worst": min_bal,
                "improvement": improvement,
                "seeds_per_candidate": seeds_per_candidate,
                "run_bal_acc": run_metrics,
                "cm_sum": cm_sum,
                "run_results": run_results,
            }
            cand["stages"].append(row)
            stage_rows.append(row)
            all_stage_rows.append(row)
            print(
                f"[adaptive] {cand['name']}: bal_mean={mean_bal:.4f}, "
                f"std={std_bal:.4f}, worst={min_bal:.4f}, "
                f"delta={improvement if np.isfinite(improvement) else float('nan'):.4f}, "
                f"epochs={epochs_local}"
            )

        stage_rows.sort(key=lambda x: x["bal_mean"], reverse=True)
        keep_n = max(1, min(len(stage_rows), max(min_keep, int(math.ceil(len(stage_rows) * keep_ratio)))))
        promoted_names = {row["name"] for row in stage_rows[:keep_n]}

        next_active = []
        for cand in active:
            latest = cand["stages"][-1]
            improved_enough = (stage_idx == 1) or (latest["improvement"] >= min_stage_improvement)
            if cand["name"] in promoted_names and improved_enough and latest["epochs"] < max_epochs:
                cand["prev_score"] = latest["bal_mean"]
                cand["epochs"] = min(max_epochs, int(math.ceil(latest["epochs"] * growth_factor)))
                next_active.append(cand)

        if len(next_active) == 0:
            print("[adaptive] Stopping: no candidates qualified for promotion.")
            break
        active = next_active

    leaderboard = sorted(all_stage_rows, key=lambda x: x["bal_mean"], reverse=True)
    best = leaderboard[0] if leaderboard else None
    return {
        "best": best,
        "leaderboard": leaderboard,
        "all_stage_rows": all_stage_rows,
        "tracked_candidates": tracked,
    }
