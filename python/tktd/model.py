"""Differentiable TKTD-IT model: multi-compound TK -> MLP bridge -> IT survival.

Line-by-line port of src/JAGS_TKTD_IT_generic.txt. Every architecture the JAGS
file can express is expressible here with the same flags, and the parameters
live on the same scale (kd_log10, W_raw, b_raw, hb_log10, beta_log10,
alpha_log10) so that JAGS posteriors and PyTorch estimates are directly
comparable.

The only deliberate departure from the JAGS source is numerical: the survival
probability is built in log space. The identities used are exact, not
approximations:

    F      = D^beta / (D^beta + alpha^beta) = sigmoid(beta * (log D - log alpha))
    1 - F  = sigmoid(-beta * (log D - log alpha))
    max_t exp(z) = exp(max_t z)

This avoids the overflow that D^beta would produce for large damage or large
beta (D can legitimately reach exp(30) and beta can reach 100), which in
floating point would give inf/inf = nan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bridge import BridgeSpec
from .data import TKTDDataset

LOG10 = math.log(10.0)
EPS = 1e-10           # JAGS: psurv and alpha are floored at 1e-10
LOG_EPS = math.log(EPS)
TINY = 1e-300         # floor before taking a log of a possibly-zero damage


@dataclass
class ModelConfig:
    """Architecture flags and priors, mirroring the JAGS data block."""

    out_exp: int = 1
    alpha_split: int = 0
    alpha_out_exp: int = 1
    alpha_activation: int = 1
    # priors (identical defaults to fit_TKTD_bayes)
    kd_meanlog10: float = -1.5
    kd_sdlog10: float = 0.5
    hb_meanlog10: float = -1.5
    hb_sdlog10: float = 0.5
    alpha_meanlog10: float = -1.5
    alpha_sdlog10: float = 0.5
    beta_minlog10: float = -2.0
    beta_maxlog10: float = 2.0
    # hb_value == 0 freezes hb at hb_valueFIXED (JAGS: ifelse(hb_value == 0, ...))
    hb_value: int = 1
    hb_valueFIXED: float = 0.0


class TensorData:
    """Dataset arrays as tensors, padded to the bridge width M."""

    def __init__(self, dataset: TKTDDataset, M: int, dtype=torch.float64, device="cpu"):
        n, n_X = dataset.n_data, dataset.n_X
        Xpad = np.zeros((n, M), dtype=np.float64)
        Xpad[:, :n_X] = dataset.X

        # torch.tensor (not as_tensor) because pandas-backed arrays can be read-only
        t = lambda a, dt=dtype: torch.tensor(a, dtype=dt, device=device)
        self.Xpad = t(Xpad)
        self.time = t(dataset.time)
        self.Nsurv = t(dataset.Nsurv)
        self.Nprec = t(dataset.Nprec)
        self.i_prec = t(dataset.i_prec, torch.long)
        self.rep_id = t(dataset.replicate_ID, torch.long)
        self.t_id = t(dataset.time_ID, torch.long)
        self.n_rep = dataset.n_replicates
        self.T_max = dataset.n_time_max
        self.n_data = n
        self.dtype = dtype
        self.device = device

        if torch.any(self.Nsurv > self.Nprec):
            raise ValueError(
                "Found Nsurv > Nprec: survival must be non-increasing within a "
                "replicate. Check that the CSV is sorted by replicate then time."
            )
        # constant term of the binomial log-pmf, so deviance matches JAGS'
        # (which includes the normalising constant)
        self.log_binom_coef = (
            torch.lgamma(self.Nprec + 1)
            - torch.lgamma(self.Nsurv + 1)
            - torch.lgamma(self.Nprec - self.Nsurv + 1)
        )


class TKTDIT(nn.Module):
    def __init__(self, bridge: BridgeSpec, cfg: ModelConfig, n_X: int,
                 dtype=torch.float64, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.bridge = bridge
        self.n_X = n_X
        self.n_layer = bridge.n_layer
        self.M = bridge.M

        buf = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=device)
        self.register_buffer("W_mask", buf(bridge.W_mask))
        self.register_buffer("b_mask", buf(bridge.b_mask))
        self.register_buffer("neg_slope", buf(bridge.neg_slope))
        self.register_buffer("in_mask", buf(bridge.in_mask))
        self.register_buffer("kd_idx", buf(bridge.kd_idx, torch.long))
        self.register_buffer("alpha_idx", buf(bridge.alpha_idx, torch.long))

        p = lambda *shape: nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))
        self.kd_log10 = p(n_X)
        self.W_raw = p(self.n_layer, self.M, self.M)
        self.b_raw = p(self.n_layer, self.M)
        self.hb_log10 = p()
        self.beta_raw = p()
        self.alpha_log10 = p(n_X)

        self.sd_w = 1.0 / math.sqrt(bridge.prec_w)
        self.hb_free = cfg.hb_value != 0

    # ---------------------------------------------------------------- params
    @property
    def beta_log10(self) -> Tensor:
        lo, hi = self.cfg.beta_minlog10, self.cfg.beta_maxlog10
        return lo + (hi - lo) * torch.sigmoid(self.beta_raw)

    @property
    def beta(self) -> Tensor:
        return torch.pow(10.0, self.beta_log10)

    @property
    def hb(self) -> Tensor:
        if not self.hb_free:
            return torch.as_tensor(self.cfg.hb_valueFIXED, dtype=self.hb_log10.dtype,
                                   device=self.hb_log10.device)
        return torch.pow(10.0, self.hb_log10)

    @property
    def kd(self) -> Tensor:
        return torch.pow(10.0, self.kd_log10)

    def n_effective_params(self) -> int:
        """Parameters actually connected to the likelihood (for AIC/BIC)."""
        n = self.n_X + self.bridge.n_free_weights + 1  # kd + W/b + beta
        n += 1 if self.hb_free else 0
        n += self.n_X if self.cfg.alpha_split else 1
        return int(n)

    @torch.no_grad()
    def init_from_prior(self, generator: torch.Generator | None = None, scale: float = 1.0,
                        input_scale: float = 1.0):
        """Draw a starting point from the JAGS priors, rescaled to stay in range.

        input_scale is the typical magnitude of the TK damage entering the
        first layer. Exposure concentrations run to ~1e3 while the weight prior
        is N(0, 0.5), so drawing the first layer straight from the prior gives
        a bridge output of ~1e2, hence a survival probability pinned at the
        1e-10 floor, where the gradient is exactly zero and the optimiser can
        never recover. Shrinking the first layer by input_scale keeps the
        initial bridge output of order 1. This changes only where the search
        starts -- the objective, priors and model are untouched.
        """
        c = self.cfg
        rn = lambda shape: torch.randn(shape, generator=generator, dtype=self.kd_log10.dtype,
                                       device=self.kd_log10.device)
        self.kd_log10.copy_(c.kd_meanlog10 + scale * c.kd_sdlog10 * rn(self.kd_log10.shape))
        self.alpha_log10.copy_(c.alpha_meanlog10 + scale * c.alpha_sdlog10 * rn(self.alpha_log10.shape))
        self.hb_log10.copy_(c.hb_meanlog10 + scale * c.hb_sdlog10 * rn(()))

        self.b_raw.copy_(scale * 0.1 * self.sd_w * rn(self.b_raw.shape))
        sizes = self.bridge.layer_sizes
        for l in range(self.n_layer):
            if l == 0:
                sd_l = self.sd_w / max(1.0, input_scale)
            else:
                sd_l = self.sd_w / math.sqrt(max(sizes[l], 1))   # He-style, keeps z ~ O(1)
            self.W_raw[l].copy_(scale * sd_l * rn(self.W_raw[l].shape))

        # beta_raw = 0 <=> beta_log10 at the middle of its uniform prior
        self.beta_raw.copy_(0.3 * rn(()))

    @torch.no_grad()
    def calibrate_output_bias(self, td: "TensorData") -> None:
        """Shift the output bias so the initial damage lands on the threshold.

        Straight from the priors, the bridge output is of order 1 while alpha
        starts at 10^-1.5, so beta * (log D - log alpha) is large and positive:
        the model predicts that everything dies, F is saturated at 1 and the
        gradient carries almost no information about which way to move.

        Shifting the last layer's bias so that the median damage equals the
        threshold starts every restart in the responsive part of the survival
        curve. The bias is a free parameter with a N(0, 0.5) prior, so this is
        only a choice of starting point, exactly like a data-dependent neural
        network initialisation.

        Skipped for 'split' alpha: there the threshold is produced by the same
        network, so the output bias shifts the damage and the threshold
        together and cancels out.
        """
        if self.cfg.alpha_split:
            return
        out = self(td)
        log_a = self.log_alpha()
        if self.cfg.out_exp:
            shift = float(log_a - out["log_D"].median())
        else:
            D = torch.exp(out["log_D"])
            shift = float(torch.exp(log_a) - D.median())
        if math.isfinite(shift):
            self.b_raw[self.n_layer - 1, 0] += shift

    # --------------------------------------------------------------- forward
    def _masked(self) -> tuple[Tensor, Tensor]:
        return self.W_raw * self.W_mask, self.b_raw * self.b_mask

    def log_alpha(self) -> Tensor:
        """log of the IT threshold, global or 'split' through the same network."""
        c = self.cfg
        if not c.alpha_split:
            return torch.clamp_min(self.alpha_log10[0] * LOG10, LOG_EPS)

        W, b = self._masked()
        ha = self.in_mask * torch.pow(10.0, self.alpha_log10[self.alpha_idx])
        for l in range(self.n_layer):
            za = W[l] @ ha + b[l]
            if l < self.n_layer - 1:
                # alpha_activation == 0 keeps the alpha branch purely affine,
                # reproducing Bayes_TKNNTD_split.R (ReLU on D, none on alpha)
                ha = torch.maximum(za, self.neg_slope[l] * za) if c.alpha_activation else za
        za_out = za[0]
        if c.alpha_out_exp:
            return torch.clamp_min(za_out, LOG_EPS)          # alpha = max(exp(za), eps)
        return torch.log(torch.clamp_min(za_out, EPS))       # alpha = max(za, eps)

    def log_damage(self, td: TensorData) -> Tensor:
        """log of D_pos: the running maximum damage up to each row's time point."""
        W, b = self._masked()
        kd_full = self.kd[self.kd_idx]                        # (M,)

        # TK under constant exposure: D_j(t) = X_j * (1 - exp(-kd_j t))
        h = td.Xpad * (-torch.expm1(-kd_full.unsqueeze(0) * td.time.unsqueeze(1)))

        for l in range(self.n_layer):
            z = h @ W[l].transpose(0, 1) + b[l]
            if l < self.n_layer - 1:
                h = torch.maximum(z, self.neg_slope[l] * z)
        z_out = z[:, 0]

        # running max over time within each replicate; padding cells sit far
        # below any real value and are never reached by a valid time index
        neg_inf = torch.finfo(z_out.dtype).min / 4
        mat = torch.full((td.n_rep, td.T_max), neg_inf, dtype=z_out.dtype, device=z_out.device)
        mat = mat.index_put((td.rep_id, td.t_id), z_out)
        run = torch.cummax(mat, dim=1).values[td.rep_id, td.t_id]

        if self.cfg.out_exp:
            return run                                        # D = exp(z) > 0
        return torch.log(torch.clamp_min(torch.clamp_min(run, 0.0), TINY))

    def forward(self, td: TensorData) -> dict[str, Tensor]:
        log_D = self.log_damage(td)
        log_a = self.log_alpha()
        beta = self.beta

        # 1 - F, with F the log-logistic CDF of the tolerance threshold
        log_1mF = F.logsigmoid(-beta * (log_D - log_a))
        log_psurv = torch.clamp_min(-self.hb * td.time + log_1mF, LOG_EPS)
        ratio = torch.exp(log_psurv - log_psurv[td.i_prec]).clamp(EPS, 1.0 - EPS)

        return {
            "log_D": log_D,
            "log_alpha": log_a,
            "log_psurv": log_psurv,
            "psurv": torch.exp(log_psurv),
            "ratio": ratio,
        }

    # ------------------------------------------------------------ objectives
    def log_lik_pointwise(self, td: TensorData, out: dict[str, Tensor] | None = None) -> Tensor:
        out = self(td) if out is None else out
        p = out["ratio"]
        return (
            td.log_binom_coef
            + td.Nsurv * torch.log(p)
            + (td.Nprec - td.Nsurv) * torch.log1p(-p)
        )

    def log_lik(self, td: TensorData, out: dict[str, Tensor] | None = None) -> Tensor:
        return self.log_lik_pointwise(td, out).sum()

    def log_prior(self) -> Tensor:
        c = self.cfg
        norm = lambda x, mu, sd: (
            -0.5 * ((x - mu) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2 * math.pi)
        ).sum()
        lp = norm(self.kd_log10, c.kd_meanlog10, c.kd_sdlog10)
        lp = lp + norm(self.alpha_log10, c.alpha_meanlog10, c.alpha_sdlog10)
        lp = lp + norm(self.W_raw, 0.0, self.sd_w) + norm(self.b_raw, 0.0, self.sd_w)
        if self.hb_free:
            lp = lp + norm(self.hb_log10, c.hb_meanlog10, c.hb_sdlog10)
        # beta_log10 ~ Uniform: constant density inside the box, which the
        # bounded reparameterisation already enforces, so it adds nothing
        return lp

    def objective(self, td: TensorData, use_prior: bool = True) -> Tensor:
        """Negative log posterior (MAP) or negative log likelihood (MLE)."""
        nll = -self.log_lik(td)
        return nll - self.log_prior() if use_prior else nll

    # --------------------------------------------------------------- reports
    @torch.no_grad()
    def parameter_report(self, compounds: list[str]) -> dict:
        alpha = float(torch.exp(self.log_alpha()))
        return {
            "kd": {c: float(v) for c, v in zip(compounds, self.kd)},
            "alpha": alpha,
            "alpha_per_compound": (
                {c: float(10.0 ** v) for c, v in zip(compounds, self.alpha_log10)}
                if self.cfg.alpha_split else None
            ),
            "beta": float(self.beta),
            "hb": float(self.hb),
            "W": (self.W_raw * self.W_mask).tolist(),
            "b": (self.b_raw * self.b_mask).tolist(),
        }
