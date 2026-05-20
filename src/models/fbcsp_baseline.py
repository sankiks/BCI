"""Classical FBCSP baselines for motor-imagery decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ClassifierType = Literal["lda", "svm"]

DEFAULT_FILTER_BANKS: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 30.0),
)


@dataclass(frozen=True)
class FBCSPConfig:
    """Configuration for filter-bank CSP baseline."""

    classifier: ClassifierType = "lda"
    filter_banks: tuple[tuple[float, float], ...] = DEFAULT_FILTER_BANKS
    csp_n_components: int = 4
    csp_reg: float | str | None = None
    csp_log: bool | None = True
    filter_order: int = 4
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"


@dataclass(frozen=True)
class BaselineMetrics:
    """Metrics for one evaluation result."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray


@dataclass(frozen=True)
class BaselineFoldResult:
    """Prediction outputs and metrics for one split."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: BaselineMetrics


def _validate_epoch_data(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError("X must have shape (n_epochs, n_channels, n_times).")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if len(X) != len(y):
        raise ValueError("X and y must have matching first dimension.")
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes for classification.")


def _validate_filter_banks(filter_banks: Sequence[tuple[float, float]], sfreq: float) -> None:
    nyquist = 0.5 * float(sfreq)
    for low, high in filter_banks:
        if not (0.0 < low < high < nyquist):
            raise ValueError(
                f"Invalid filter bank ({low}, {high}) for sfreq={sfreq}. "
                f"Require 0 < low < high < {nyquist}."
            )


def _bandpass_epochs(X: np.ndarray, sfreq: float, low: float, high: float, order: int) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


class FBCSPClassifier:
    """Filter-Bank CSP feature extractor plus classifier."""

    def __init__(self, sfreq: float, config: FBCSPConfig | None = None) -> None:
        self.sfreq = float(sfreq)
        self.config = config or FBCSPConfig()
        _validate_filter_banks(self.config.filter_banks, self.sfreq)
        self._csp_models: list[CSP] = []
        self._clf = None

    def _make_classifier(self):
        if self.config.classifier == "lda":
            return LinearDiscriminantAnalysis()
        if self.config.classifier == "svm":
            return make_pipeline(
                StandardScaler(),
                SVC(
                    C=self.config.svm_c,
                    kernel=self.config.svm_kernel,
                    gamma=self.config.svm_gamma,
                    class_weight="balanced",
                ),
            )
        raise ValueError(f"Unsupported classifier: {self.config.classifier}")

    def _fit_csp_feature_stack(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._csp_models = []
        band_features: list[np.ndarray] = []
        for low, high in self.config.filter_banks:
            X_band = _bandpass_epochs(
                X=X,
                sfreq=self.sfreq,
                low=low,
                high=high,
                order=self.config.filter_order,
            )
            csp = CSP(
                n_components=self.config.csp_n_components,
                reg=self.config.csp_reg,
                log=self.config.csp_log,
                transform_into="average_power",
                norm_trace=False,
            )
            features = csp.fit_transform(X_band, y)
            self._csp_models.append(csp)
            band_features.append(features)
        return np.concatenate(band_features, axis=1)

    def _transform_csp_feature_stack(self, X: np.ndarray) -> np.ndarray:
        if not self._csp_models:
            raise RuntimeError("FBCSPClassifier is not fitted yet.")
        band_features: list[np.ndarray] = []
        for (low, high), csp in zip(self.config.filter_banks, self._csp_models):
            X_band = _bandpass_epochs(
                X=X,
                sfreq=self.sfreq,
                low=low,
                high=high,
                order=self.config.filter_order,
            )
            band_features.append(csp.transform(X_band))
        return np.concatenate(band_features, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPClassifier":
        _validate_epoch_data(X, y)
        features = self._fit_csp_feature_stack(X, y)
        self._clf = self._make_classifier()
        self._clf.fit(features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("FBCSPClassifier is not fitted yet.")
        features = self._transform_csp_feature_stack(X)
        return self._clf.predict(features)


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> BaselineMetrics:
    """Compute required benchmark metrics."""
    metric_labels = labels if labels is not None else np.unique(y_true)
    return BaselineMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=metric_labels),
    )


def evaluate_fbcsp_split(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    config: FBCSPConfig | None = None,
    fold: int = 0,
    label_order: Sequence[int] | None = None,
) -> BaselineFoldResult:
    """Train on one split and return predictions + metrics."""
    _validate_epoch_data(X, y)
    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    test_idx_array = np.asarray(test_idx, dtype=np.int64)
    if len(train_idx_array) == 0 or len(test_idx_array) == 0:
        raise ValueError("train_idx and test_idx must both be non-empty.")

    model = FBCSPClassifier(sfreq=sfreq, config=config)
    model.fit(X[train_idx_array], y[train_idx_array])
    y_pred = model.predict(X[test_idx_array])
    y_true = y[test_idx_array]

    labels = None if label_order is None else np.asarray(label_order, dtype=np.int64)
    metrics = compute_baseline_metrics(y_true=y_true, y_pred=y_pred, labels=labels)
    return BaselineFoldResult(
        fold=int(fold),
        train_idx=train_idx_array,
        test_idx=test_idx_array,
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
    )


def evaluate_fbcsp_splits(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    splits: Sequence[object],
    config: FBCSPConfig | None = None,
    label_order: Sequence[int] | None = None,
) -> list[BaselineFoldResult]:
    """Evaluate FBCSP baseline over multiple split objects.

    Each split object must expose `train_idx`, `test_idx`, and optionally `fold`.
    """
    results: list[BaselineFoldResult] = []
    for idx, split in enumerate(splits):
        if not hasattr(split, "train_idx") or not hasattr(split, "test_idx"):
            raise ValueError("Each split must expose `train_idx` and `test_idx`.")
        fold = int(getattr(split, "fold", idx))
        results.append(
            evaluate_fbcsp_split(
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


def summarize_baseline_results(results: Sequence[BaselineFoldResult]) -> dict[str, float]:
    """Summarize mean/std over folds for scalar metrics."""
    if len(results) == 0:
        raise ValueError("No fold results provided.")
    accuracy = np.array([result.metrics.accuracy for result in results], dtype=float)
    balanced_accuracy = np.array([result.metrics.balanced_accuracy for result in results], dtype=float)
    macro_f1 = np.array([result.metrics.macro_f1 for result in results], dtype=float)
    return {
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std(ddof=0)),
        "balanced_accuracy_mean": float(balanced_accuracy.mean()),
        "balanced_accuracy_std": float(balanced_accuracy.std(ddof=0)),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std(ddof=0)),
    }

