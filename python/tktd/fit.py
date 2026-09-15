"""Point-estimate fitting (MAP / MLE) with an optional Laplace posterior.

Replaces the MCMC step of the JAGS pipeline. The model is fully differentiable,
so the posterior mode is reached by gradient descent in seconds instead of
minutes of Gibbs sampling.

Two fitting modes:
  - MAP (default): maximises log-likelihood + log-prior, using exactly the
    priors declared in the JAGS file. The optimum is the posterior mode, which
    is the point estimate most directly comparable to a JAGS posterior mean.
  - MLE (--no-prior): maximises the likelihood alone. Kept for reference, but
    the bridge weights are only weakly identified, so MLE tends to drift and
    its Hessian is singular in the masked ("phantom neuron") directions.

The Laplace step approximates the posterior by a Gaussian at the mode, with
covariance equal to the inverse Hessian of the negative log posterior. Drawing
from it recovers the two quantities JAGS reports (posterior-mean deviance and
WAIC) so the two pipelines can be compared on the same footing.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .bridge import BridgeSpec, build_bridge
from .data import TKTDDataset
from .model import ModelConfig, TensorData, TKTDIT


class _ObjectiveModule(nn.Module):
    """Wrapper so the objective can be called as a pure function of parameters."""

    def __init__(self, model: TKTDIT, td: TensorData, use_prior: bool):
        super().__init__()
        self.model = model
        self._td = td
        self._use_prior = use_prior

    def forward(self) -> Tensor:
        return self.model.objective(self._td, self._use_prior)


class _PointwiseModule(nn.Module):
    def __init__(self, model: TKTDIT, td: TensorData):
        super().__init__()
        self.model = model
        self._td = td

    def forward(self) -> Tensor:
        return self.model.log_lik_pointwise(self._td)


def _flat_params(module: nn.Module) -> tuple[list[str], Tensor]:
    names, chunks = [], []
    for n, p in module.named_parameters():
        names.append(n)
        chunks.append(p.detach().reshape(-1))
    return names, torch.cat(chunks)


def _unflatten(module: nn.Module, flat: Tensor) -> dict[str, Tensor]:
    out, i = {}, 0
    for n, p in module.named_parameters():
        k = p.numel()
        out[n] = flat[i : i + k].view(p.shape)
        i += k
    return out


def _typical_damage_scale(dataset: TKTDDataset, cfg: ModelConfig) -> float:
    """Magnitude of the TK damage entering the bridge, at the kd prior mean."""
    import numpy as np

    kd0 = 10.0 ** cfg.kd_meanlog10
    h0 = dataset.X * (1.0 - np.exp(-kd0 * dataset.time[:, None]))
    nonzero = np.abs(h0[h0 != 0.0])
    return float(np.percentile(nonzero, 95)) if nonzero.size else 1.0


@dataclass
class FitResult:
    model: TKTDIT
    td: TensorData
    objective: float
    loglik: float
    deviance: float
    n_params: int
    aic: float
    bic: float
    converged: bool
    n_restarts_ok: int
    restart_objectives: list[float] = field(default_factory=list)
    laplace: dict | None = None


def _run_optimizer(model: TKTDIT, td: TensorData, use_prior: bool,
                   adam_steps: int, adam_lr: float, lbfgs_steps: int) -> float:
    """Adam for a robust descent, then L-BFGS to polish the mode."""
    adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    sched = torch.optim.lr_scheduler.StepLR(adam, step_size=max(adam_steps // 3, 1), gamma=0.3)
    for _ in range(adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = model.objective(td, use_prior)
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e4)
        adam.step()
        sched.step()

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=lbfgs_steps, history_size=50,
            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss = model.objective(td, use_prior)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        before = float(model.objective(td, use_prior).detach())
        try:
            lbfgs.step(closure)
        except Exception as exc:
            # The strong-Wolfe line search can degenerate on this model's flat
            # scaling ridge (PyTorch raises IndexError as well as RuntimeError).
            # L-BFGS only polishes an already-good Adam solution, so fall back
            # to that rather than losing the restart.
            warnings.warn(f"L-BFGS stopped early ({type(exc).__name__}: {exc})", stacklevel=2)
            model.load_state_dict(state)
        with torch.no_grad():
            after = model.objective(td, use_prior)
        if not torch.isfinite(after) or float(after) > before:
            model.load_state_dict(state)

    with torch.no_grad():
        final = model.objective(td, use_prior)
    return float(final) if torch.isfinite(final) else float("inf")


def laplace_posterior(model: TKTDIT, td: TensorData, use_prior: bool = True,
                      n_samples: int = 2000, seed: int = 0) -> dict | None:
    """Gaussian approximation at the mode: standard errors, and WAIC when valid.

    Standard errors come from the inverse Hessian and are always reported.

    The WAIC/posterior-deviance part is gated behind a quality check, because
    this model has an exact flat direction in the likelihood: scaling the last
    layer's weights and biases by c, log(alpha) by c and beta by 1/c leaves
    every predicted survival probability unchanged. Only the prior pins that
    ridge down, so the posterior is markedly non-Gaussian along it and a naive
    Laplace sample can wander into regions of absurdly low likelihood. When
    that happens the WAIC fields are returned as None with a reason rather than
    as a plausible-looking but meaningless number.

    Returns None when the Hessian cannot be made positive definite at all.
    """
    obj_mod = _ObjectiveModule(model, td, use_prior)
    names, flat0 = _flat_params(obj_mod)

    def f(flat: Tensor) -> Tensor:
        return functional_call(obj_mod, _unflatten(obj_mod, flat), ())

    H = torch.autograd.functional.hessian(f, flat0.clone().requires_grad_(True))
    H = 0.5 * (H + H.transpose(0, 1))  # symmetrise away numerical asymmetry

    scale = float(torch.diagonal(H).abs().mean()) or 1.0
    chol = None
    for jitter_exp in range(-10, 1):
        Hj = H + torch.eye(H.shape[0], dtype=H.dtype) * (scale * 10.0 ** jitter_exp)
        try:
            chol = torch.linalg.cholesky(Hj)
            H = Hj
            break
        except RuntimeError:
            continue
    if chol is None:
        return None

    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn(n_samples, flat0.numel(), dtype=flat0.dtype, generator=gen)
    # x = mu + L^{-T} eps  has covariance H^{-1} when H = L L^T
    draws = flat0.unsqueeze(0) + torch.linalg.solve_triangular(
        chol.transpose(0, 1), eps.transpose(0, 1), upper=True
    ).transpose(0, 1)

    pw_mod = _PointwiseModule(model, td)
    with torch.no_grad():
        ll = torch.stack([
            functional_call(pw_mod, _unflatten(pw_mod, draws[s]), ())
            for s in range(n_samples)
        ])                                            # (S, n_data)

    # --- standard errors from the inverse Hessian (always reported)
    cov = torch.linalg.inv(H)
    sd_flat = torch.sqrt(torch.clamp_min(torch.diagonal(cov), 0.0))
    sizes = [p.numel() for _, p in obj_mod.named_parameters()]
    sd_by_name = dict(zip(names, torch.split(sd_flat, sizes)))
    get = lambda key: sd_by_name.get(f"model.{key}")

    # beta is optimised through a sigmoid; map its SE back to log10(beta)
    lo, hi = model.cfg.beta_minlog10, model.cfg.beta_maxlog10
    with torch.no_grad():
        dbeta = float((hi - lo) * torch.sigmoid(model.beta_raw) * (1 - torch.sigmoid(model.beta_raw)))
    se = {
        "kd_log10": get("kd_log10").tolist(),
        "alpha_log10": get("alpha_log10").tolist(),
        "hb_log10": float(get("hb_log10")[0]),
        # a saturated sigmoid makes the delta-method SE collapse to ~0, which
        # would read as false precision; report nothing rather than that
        "beta_log10": (float(get("beta_raw")[0]) * dbeta) if dbeta > 1e-8 else None,
    }

    result = {"n_samples_requested": int(n_samples), "standard_errors": se}

    # --- WAIC, only if the Gaussian approximation actually holds here
    finite = torch.isfinite(ll).all(dim=1)
    ll = ll[finite]
    with torch.no_grad():
        ll_mode = float(model.log_lik(td))

    if ll.shape[0] < max(50, n_samples // 10):
        result.update(waic_total=None, waic_valid=False,
                      waic_invalid_reason="too many non-finite posterior draws")
        return result

    S = ll.shape[0]
    total_ll = ll.sum(dim=1)
    deviance_mode = -2.0 * ll_mode
    dev_mean = float((-2.0 * total_ll).mean())
    k = max(model.n_effective_params(), 1)

    # Under a Gaussian posterior the mean deviance exceeds the deviance at the
    # mode by the effective number of parameters. Far more than that means the
    # draws are leaving the region the quadratic approximation describes --
    # which is what this model's scaling ridge does -- so the WAIC computed
    # from them would be meaningless.
    if dev_mean - deviance_mode > 10.0 * k:
        result.update(
            waic_total=None, waic_valid=False,
            waic_invalid_reason=(
                f"Laplace draws leave the quadratic region: mean deviance "
                f"{dev_mean:.1f} vs {deviance_mode:.1f} at the mode, a gap of "
                f"{dev_mean - deviance_mode:.1f} against an expected ~{k}. The "
                f"posterior is too non-Gaussian along the model's scaling ridge. "
                f"Use AIC/BIC for model comparison."
            ),
        )
        return result

    lppd = (torch.logsumexp(ll, dim=0) - math.log(S)).sum()
    p_waic = ll.var(dim=0, unbiased=True).sum()
    waic = -2.0 * (lppd - p_waic)
    deviance_draws = -2.0 * total_ll

    result.update(
        n_samples_used=int(S),
        waic_valid=True,
        deviance_posterior_mean=float(deviance_draws.mean()),
        deviance_posterior_sd=float(deviance_draws.std()),
        lppd=float(lppd),
        p_waic=float(p_waic),
        waic_total=float(waic),
        waic_per_obs=float(waic / td.n_data),
    )
    return result


def fit_tktd(
    dataset: TKTDDataset,
    hidden_layers=(),
    neg_slope: float = 0.0,
    cfg: ModelConfig | None = None,
    prec_w: float = 4.0,
    n_restarts: int = 12,
    adam_steps: int = 3000,
    adam_lr: float = 0.2,
    lbfgs_steps: int = 500,
    seed: int = 0,
    use_prior: bool = True,
    laplace: bool = False,
    laplace_samples: int = 2000,
    verbose: bool = True,
) -> FitResult:
    cfg = cfg or ModelConfig()
    bridge = build_bridge(dataset.n_X, hidden_layers, neg_slope, prec_w)
    td = TensorData(dataset, bridge.M)
    input_scale = _typical_damage_scale(dataset, cfg)

    best_state, best_obj, objectives = None, float("inf"), []
    for r in range(n_restarts):
        model = TKTDIT(bridge, cfg, dataset.n_X)
        gen = torch.Generator().manual_seed(seed + 1000 * r)
        model.init_from_prior(gen, input_scale=input_scale)
        # The objective is genuinely multi-modal, so restarts alternate between
        # a threshold-calibrated and a raw prior start: each wins on different
        # architectures, and mixing them widens the search.
        if r % 2 == 0:
            model.calibrate_output_bias(td)
        obj = _run_optimizer(model, td, use_prior, adam_steps, adam_lr, lbfgs_steps)
        objectives.append(obj)
        if obj < best_obj:
            best_obj, best_state = obj, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  restart {r + 1}/{n_restarts}: objective = {obj:.4f}", flush=True)

    n_ok = sum(1 for o in objectives if math.isfinite(o))
    if best_state is None or not math.isfinite(best_obj):
        raise RuntimeError("Every restart diverged; try more restarts or a smaller learning rate.")

    model = TKTDIT(bridge, cfg, dataset.n_X)
    model.load_state_dict(best_state)

    with torch.no_grad():
        loglik = float(model.log_lik(td))
    deviance = -2.0 * loglik
    k = model.n_effective_params()

    lap = None
    if laplace:
        if not use_prior:
            warnings.warn(
                "Laplace approximation skipped: without priors the Hessian is "
                "singular in the masked-weight directions. Re-run in MAP mode.",
                stacklevel=2,
            )
        else:
            lap = laplace_posterior(model, td, use_prior, laplace_samples, seed)
            if lap is None and verbose:
                print("  Laplace approximation failed (Hessian not positive definite)", flush=True)

    # a restart is 'converged' if several independent starts reached the same optimum
    tol = 1e-3 * max(1.0, abs(best_obj))
    agree = sum(1 for o in objectives if math.isfinite(o) and abs(o - best_obj) < tol)

    return FitResult(
        model=model,
        td=td,
        objective=best_obj,
        loglik=loglik,
        deviance=deviance,
        n_params=k,
        aic=deviance + 2 * k,
        bic=deviance + k * math.log(td.n_data),
        converged=agree >= 2,
        n_restarts_ok=n_ok,
        restart_objectives=objectives,
        laplace=lap,
    )


@torch.no_grad()
def predictions_frame(result: FitResult, dataset: TKTDDataset):
    """Per-row fitted quantities, for comparing curves against the JAGS fit."""
    import pandas as pd

    out = result.model(result.td)
    return pd.DataFrame({
        "replicate": [dataset.replicates[i] for i in dataset.replicate_ID],
        "time": dataset.time,
        "Nprec": dataset.Nprec,
        "Nsurv_obs": dataset.Nsurv,
        "ratio": out["ratio"].numpy(),
        "psurv": out["psurv"].numpy(),
        "Nsurv_expected": (dataset.Nprec * out["ratio"].numpy()),
        "D_max": torch.exp(out["log_D"]).numpy(),
    })
