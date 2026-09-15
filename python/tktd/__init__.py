"""Differentiable PyTorch implementation of the TKTD-IT neural-bridge model.

Mirrors src/JAGS_TKTD_IT_generic.txt and R/Bayes_TK-NN-TD_generic.R.
"""

from .architectures import ARCHITECTURES, ARTIFICIAL_DATASETS, REAL_DATASETS, Architecture
from .bridge import BridgeSpec, build_bridge
from .data import TKTDDataset, infer_mixture_columns, load_tktd_csv
from .fit import FitResult, fit_tktd, laplace_posterior, predictions_frame
from .model import ModelConfig, TensorData, TKTDIT

__all__ = [
    "ARCHITECTURES", "ARTIFICIAL_DATASETS", "REAL_DATASETS", "Architecture",
    "BridgeSpec", "build_bridge",
    "TKTDDataset", "infer_mixture_columns", "load_tktd_csv",
    "FitResult", "fit_tktd", "laplace_posterior", "predictions_frame",
    "ModelConfig", "TensorData", "TKTDIT",
]
