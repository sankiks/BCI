from .cnn_transformer_variant import (
    CNNTransformerConfig,
    CNNTransformerEEG,
    build_cnn_transformer_eeg,
    compare_cnn_vs_transformer,
)
from .eegnet_baseline import (
    EEGNetConfig,
    EEGNetFoldResult,
    build_eegnet_model,
    summarize_eegnet_results,
    train_eegnet_split,
    train_eegnet_splits,
)
from .fbcsp_baseline import (
    DEFAULT_FILTER_BANKS,
    BaselineFoldResult,
    BaselineMetrics,
    FBCSPClassifier,
    FBCSPConfig,
    compute_baseline_metrics,
    evaluate_fbcsp_split,
    evaluate_fbcsp_splits,
    summarize_baseline_results,
)
from .multi_scale_eeg_cnn import (
    MultiScaleEEGCNN,
    MultiScaleEEGCNNConfig,
    build_multiscale_eeg_cnn,
)

__all__ = [
    "CNNTransformerConfig",
    "CNNTransformerEEG",
    "build_cnn_transformer_eeg",
    "compare_cnn_vs_transformer",
    "EEGNetConfig",
    "EEGNetFoldResult",
    "build_eegnet_model",
    "summarize_eegnet_results",
    "train_eegnet_split",
    "train_eegnet_splits",
    "DEFAULT_FILTER_BANKS",
    "BaselineFoldResult",
    "BaselineMetrics",
    "FBCSPClassifier",
    "FBCSPConfig",
    "compute_baseline_metrics",
    "evaluate_fbcsp_split",
    "evaluate_fbcsp_splits",
    "summarize_baseline_results",
    "MultiScaleEEGCNN",
    "MultiScaleEEGCNNConfig",
    "build_multiscale_eeg_cnn",
]
