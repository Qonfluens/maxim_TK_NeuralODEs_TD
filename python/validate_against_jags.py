"""Step 2 of the JAGS <-> PyTorch equivalence check.

Loads the dumps written by validate_against_jags.R, injects JAGS' own parameter
values into the PyTorch model, and checks that both compute the same deviance.

This is a much stronger test than comparing two independent fits: it isolates
the model definition from the inference algorithm. If the deviances agree to
machine precision, the port is exact and any remaining difference between the
two pipelines comes from the estimation method (MCMC vs gradient descent), not
from the model.

Usage (from the repo root):
    Rscript python/validate_against_jags.R output/validation
    python3 python/validate_against_jags.py output/validation
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tktd.bridge import build_bridge
from tktd.data import load_tktd_csv
from tktd.model import ModelConfig, TensorData, TKTDIT

DATA_PATH = "data/data_artificial_additive.csv"
MIXTURE = ["A", "B"]
TOL_DEVIANCE = 1e-6


def parse_indexed(name: str) -> tuple[str, tuple[int, ...]]:
    """'W[1,2,3]' -> ('W', (0, 1, 2)); 'hb' -> ('hb', ())."""
    m = re.match(r"^([A-Za-z_]+)(?:\[([0-9,]+)\])?$", name)
    if not m:
        raise ValueError(f"Unparsable monitor name: {name}")
    base, idx = m.group(1), m.group(2)
    return base, tuple(int(i) - 1 for i in idx.split(",")) if idx else ()


def load_case(out_dir: Path, case: str):
    cfg_rows = pd.read_csv(out_dir / f"cfg_{case}.csv")
    cfg_map = dict(zip(cfg_rows["key"], cfg_rows["value"]))
    params = pd.read_csv(out_dir / f"params_{case}.csv")
    data = pd.read_csv(out_dir / f"data_{case}.csv")
    return cfg_map, params, data


def check_preprocessing(data_r: pd.DataFrame, ds) -> list[str]:
    """R's data_IT is 1-based; the Python dataset is 0-based."""
    problems = []
    checks = {
        "time": (data_r["time"].to_numpy(), ds.time),
        "Nsurv": (data_r["Nsurv"].to_numpy(), ds.Nsurv),
        "Nprec": (data_r["Nprec"].to_numpy(), ds.Nprec),
        "i_prec": (data_r["i_prec"].to_numpy(), ds.i_prec + 1),
        "time_ID": (data_r["time_ID"].to_numpy(), ds.time_ID + 1),
        "replicate_ID": (data_r["replicate_ID"].to_numpy(), ds.replicate_ID + 1),
    }
    for field, (r_vals, py_vals) in checks.items():
        if not np.allclose(r_vals, py_vals):
            n_bad = int((r_vals != py_vals).sum())
            problems.append(f"{field}: {n_bad}/{len(r_vals)} rows differ")
    return problems


def run_case(out_dir: Path, case: str, n_hidden: int) -> bool:
    cfg_map, params, data_r = load_case(out_dir, case)

    ds = load_tktd_csv(DATA_PATH, MIXTURE)
    pre_problems = check_preprocessing(data_r, ds)

    cfg = ModelConfig(
        out_exp=int(cfg_map["out_exp"]),
        alpha_split=int(cfg_map["alpha_split"]),
        alpha_out_exp=int(cfg_map["alpha_out_exp"]),
        alpha_activation=int(cfg_map["alpha_activation"]),
    )
    bridge = build_bridge(
        n_X=ds.n_X,
        hidden_layers=tuple([ds.n_X] * n_hidden),
        neg_slope=float(cfg_map["neg_slope_1"]),
        prec_w=float(cfg_map["prec_w"]),
    )
    assert bridge.n_layer == int(cfg_map["n_layer"]), "layer count mismatch"
    assert bridge.M == int(cfg_map["M"]), "width mismatch"

    model = TKTDIT(bridge, cfg, ds.n_X)
    td = TensorData(ds, bridge.M)

    jags_deviance = None
    with torch.no_grad():
        for name, value in zip(params["name"], params["value"]):
            base, idx = parse_indexed(name)
            if base == "W":
                model.W_raw[idx] = value
            elif base == "b":
                model.b_raw[idx] = value
            elif base == "kd":
                model.kd_log10[idx[0]] = math.log10(value)
            elif base == "hb":
                model.hb_log10.fill_(math.log10(value))
            elif base == "alpha":
                model.alpha_log10[0] = math.log10(value)
            elif base == "beta":
                lo, hi = cfg.beta_minlog10, cfg.beta_maxlog10
                u = (math.log10(value) - lo) / (hi - lo)
                model.beta_raw.fill_(math.log(u / (1.0 - u)))
            elif base == "deviance":
                jags_deviance = float(value)
            else:
                raise ValueError(f"Unexpected monitored node: {name}")

    if jags_deviance is None:
        raise RuntimeError(f"No deviance found in params_{case}.csv")

    with torch.no_grad():
        torch_deviance = float(-2.0 * model.log_lik(td))

    diff = abs(torch_deviance - jags_deviance)
    rel = diff / max(1.0, abs(jags_deviance))
    ok = rel < TOL_DEVIANCE and not pre_problems

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {case:<28} JAGS={jags_deviance:14.8f}  "
          f"torch={torch_deviance:14.8f}  |diff|={diff:.3e}")
    for p in pre_problems:
        print(f"         preprocessing mismatch -> {p}")
    return ok


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output/validation")
    if not out_dir.is_dir():
        print(f"Validation dir not found: {out_dir}\n"
              f"Run: Rscript python/validate_against_jags.R {out_dir}")
        return 2

    cases = [("n_exp", 0), ("n", 0), ("nn_ReLU_n_exp", 1),
             ("nn_n_exp", 1), ("nn_ReLU_nn_ReLU_n_exp", 2)]
    results = []
    for case, n_hidden in cases:
        if not (out_dir / f"cfg_{case}.csv").exists():
            print(f"[SKIP] {case}: no dump found")
            continue
        results.append(run_case(out_dir, case, n_hidden))

    print()
    if results and all(results):
        print(f"All {len(results)} architectures match JAGS to < {TOL_DEVIANCE:g} "
              f"relative deviance. The PyTorch port is exact.")
        return 0
    print("Some architectures do not match; the port is NOT equivalent.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
