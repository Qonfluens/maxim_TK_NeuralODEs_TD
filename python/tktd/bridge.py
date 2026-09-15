"""Connectivity masks for the generic MLP bridge.

Port of .build_generic_bridge_data() in R/Bayes_TK-NN-TD_generic.R. The masked
("phantom neuron") formulation is kept rather than using real per-layer weight
matrices, so that the PyTorch parameterisation matches the JAGS one entry for
entry and the two can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BridgeSpec:
    M: int                  # shared width (max over all layer sizes)
    n_layer: int            # number of affine layers (1 = direct linear bridge)
    W_mask: np.ndarray      # (n_layer, M, M)
    b_mask: np.ndarray      # (n_layer, M)
    neg_slope: np.ndarray   # (n_layer,) activation slope for negative inputs
    kd_idx: np.ndarray      # (M,) 0-based index into kd[0:n_X]
    alpha_idx: np.ndarray   # (M,) 0-based index into alpha_log10[0:n_X]
    in_mask: np.ndarray     # (M,) 1 for the n_X real exposure columns, else 0
    prec_w: float
    layer_sizes: list[int]

    @property
    def n_free_weights(self) -> int:
        """Weights and biases that are actually connected to the likelihood."""
        return int(self.W_mask.sum() + self.b_mask.sum())


def build_bridge(
    n_X: int,
    hidden_layers: Sequence[int] = (),
    neg_slope: float | Sequence[float] = 0.0,
    prec_w: float = 4.0,
) -> BridgeSpec:
    layer_sizes = [int(n_X), *(int(h) for h in hidden_layers), 1]
    n_layer = len(layer_sizes) - 1
    M = max(layer_sizes)

    slope = np.asarray(neg_slope, dtype=np.float64).ravel()
    slope = np.resize(slope, n_layer)

    W_mask = np.zeros((n_layer, M, M), dtype=np.float64)
    b_mask = np.zeros((n_layer, M), dtype=np.float64)
    for l in range(n_layer):
        n_in, n_out = layer_sizes[l], layer_sizes[l + 1]
        W_mask[l, :n_out, :n_in] = 1.0
        b_mask[l, :n_out] = 1.0

    # Columns n_X..M-1 are zeroed out by in_mask / the zero-padded X, so the
    # index they point at is irrelevant; pad with 0 (R padded with 1-based 1).
    pad = lambda x: np.concatenate([x, np.zeros(M - n_X, dtype=np.int64)]) if M > n_X else x
    kd_idx = pad(np.arange(n_X, dtype=np.int64))
    alpha_idx = pad(np.arange(n_X, dtype=np.int64))
    in_mask = np.concatenate([np.ones(n_X), np.zeros(M - n_X)]).astype(np.float64)

    return BridgeSpec(
        M=M,
        n_layer=n_layer,
        W_mask=W_mask,
        b_mask=b_mask,
        neg_slope=slope,
        kd_idx=kd_idx,
        alpha_idx=alpha_idx,
        in_mask=in_mask,
        prec_w=float(prec_w),
        layer_sizes=layer_sizes,
    )
