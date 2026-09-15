"""One fit, end to end: load -> fit -> save JSON + predictions -> log to MLflow.

Shared by the single-run CLI (run_tktd_torch.py) and the batch driver
(reproduce_paper_torch.py) so both produce identically structured output.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import torch

from .architectures import Architecture
from .data import load_tktd_csv
from .fit import fit_tktd, predictions_frame
from .model import ModelConfig


def out_tag(n_layer: int, M: int, cfg: ModelConfig) -> str:
    """Mirrors the R engine's naming so JAGS and torch outputs pair up."""
    return (f"L{n_layer}_M{M}_out{cfg.out_exp}"
            f"_alphasplit{cfg.alpha_split}_aact{cfg.alpha_activation}")


def _log_to_mlflow(experiment: str, run_name: str, params: dict,
                   metrics: dict, artifacts: list[Path], verbose: bool = True) -> None:
    """Best-effort tracking: never allowed to break or delay a completed fit."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            for path in artifacts:
                if path.exists():
                    mlflow.log_artifact(str(path))
        if verbose:
            print(f"  MLflow: logged '{run_name}' to '{experiment}'", flush=True)
    except Exception as exc:
        warnings.warn(f"MLflow logging failed (the fit itself is unaffected): {exc}", stacklevel=2)


def run_fit(
    data_path: str,
    mixture_cols: list[str] | None = None,
    *,
    run_id: str | None = None,
    hidden_layers: tuple[int, ...] | None = None,
    arch: Architecture | None = None,
    neg_slope: float = 0.0,
    out_exp: int = 1,
    alpha_split: int = 0,
    alpha_out_exp: int = 1,
    alpha_activation: int = 1,
    prec_w: float = 4.0,
    n_restarts: int = 12,
    adam_steps: int = 3000,
    adam_lr: float = 0.2,
    lbfgs_steps: int = 500,
    seed: int = 0,
    use_prior: bool = True,
    laplace: bool = False,
    laplace_samples: int = 2000,
    output_dir: str = "output_torch",
    save_predictions: bool = True,
    mlflow_experiment: str = "TKTD-NeuralODEs-torch",
    verbose: bool = True,
) -> dict:
    dataset = load_tktd_csv(data_path, mixture_cols)

    if arch is not None:
        hidden_layers = arch.hidden_layers(dataset.n_X)
        neg_slope = arch.neg_slope
        out_exp, alpha_split = arch.out_exp, arch.alpha_split
        alpha_out_exp, alpha_activation = arch.alpha_out_exp, arch.alpha_activation
        arch_name = arch.name
    else:
        hidden_layers = tuple(hidden_layers or ())
        arch_name = "custom"

    cfg = ModelConfig(out_exp=out_exp, alpha_split=alpha_split,
                      alpha_out_exp=alpha_out_exp, alpha_activation=alpha_activation)
    run_id = run_id or Path(data_path).stem

    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] START {run_id} / {arch_name} "
              f"(n_X={dataset.n_X}, hidden={list(hidden_layers)})", flush=True)

    t0 = time.time()
    result = fit_tktd(
        dataset, hidden_layers=hidden_layers, neg_slope=neg_slope, cfg=cfg, prec_w=prec_w,
        n_restarts=n_restarts, adam_steps=adam_steps, adam_lr=adam_lr,
        lbfgs_steps=lbfgs_steps, seed=seed, use_prior=use_prior,
        laplace=laplace, laplace_samples=laplace_samples, verbose=False,
    )
    elapsed = time.time() - t0

    tag = out_tag(result.model.n_layer, result.model.M, cfg)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    base = out_path / f"torch_{run_id}_{tag}"

    summary = {
        "run_id": run_id,
        "architecture": arch_name,
        "data": data_path,
        "compounds": dataset.compounds,
        "n_data": dataset.n_data,
        "n_replicates": dataset.n_replicates,
        "config": {
            "hidden_layers": list(hidden_layers), "n_layer": result.model.n_layer,
            "M": result.model.M, "neg_slope": neg_slope, "out_exp": out_exp,
            "alpha_split": alpha_split, "alpha_out_exp": alpha_out_exp,
            "alpha_activation": alpha_activation, "prec_w": prec_w,
            "estimator": "MAP" if use_prior else "MLE",
        },
        "fit": {
            "objective": result.objective,
            "loglik": result.loglik,
            "deviance": result.deviance,
            "n_params": result.n_params,
            "aic": result.aic,
            "bic": result.bic,
            "converged": result.converged,
            "n_restarts": n_restarts,
            "n_restarts_finite": result.n_restarts_ok,
            "restart_objectives": result.restart_objectives,
            "seconds": elapsed,
        },
        "parameters": result.model.parameter_report(dataset.compounds),
        "laplace": result.laplace,
    }

    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))

    artifacts = [json_path]
    if save_predictions:
        pred_path = Path(f"{base}_predictions.csv")
        predictions_frame(result, dataset).to_csv(pred_path, index=False)
        artifacts.append(pred_path)

    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] DONE  {run_id} / {arch_name} "
              f"deviance={result.deviance:.2f} AIC={result.aic:.2f} "
              f"({elapsed:.1f}s) -> {json_path}", flush=True)

    metrics = {
        "deviance": result.deviance, "loglik": result.loglik,
        "aic": result.aic, "bic": result.bic, "n_params": result.n_params,
        "seconds": elapsed, "converged": float(result.converged),
    }
    if result.laplace and result.laplace.get("waic_valid"):
        metrics["waic_total"] = result.laplace["waic_total"]
        metrics["waic_per_obs"] = result.laplace["waic_per_obs"]
    _log_to_mlflow(
        mlflow_experiment, f"{run_id}_{arch_name}",
        {"architecture": arch_name, "run_id": run_id, "n_exposure": dataset.n_X,
         **summary["config"]},
        metrics, artifacts, verbose=verbose,
    )

    return summary


def set_thread_budget(threads: int = 1) -> None:
    """One thread per process: the batch driver gets parallelism from processes."""
    torch.set_num_threads(max(1, threads))
