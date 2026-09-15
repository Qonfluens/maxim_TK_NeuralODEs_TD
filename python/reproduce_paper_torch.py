#!/usr/bin/env python3
"""All 47 paper runs in the PyTorch framework: the counterpart of
run/reproduce_paper.sh.

Same job list as the JAGS pipeline:
  - the 4 real MaXim data sets x the 8 architectures of Table 1 plus the two
    'split' alpha variants of Table 3   -> 32 runs
  - the 3 artificial data sets x the 5 architectures without splitting -> 15 runs

Unlike the shell driver, this needs no Docker and no GNU parallel: the runs are
dispatched over a process pool, one torch thread per worker so the processes do
not fight over cores.

Examples (from the repo root):

    python3 python/reproduce_paper_torch.py                       # all 47
    python3 python/reproduce_paper_torch.py --jobs 48
    python3 python/reproduce_paper_torch.py --scope artificial    # 15 runs
    python3 python/reproduce_paper_torch.py --restarts 4 --adam-steps 500  # smoke test

With MLFLOW_TRACKING_URI set, every run is logged to MLflow exactly as the
JAGS pipeline is:

    export MLFLOW_TRACKING_URI=http://172.17.0.1:5000
    python3 python/reproduce_paper_torch.py --jobs 48
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tktd.architectures import ARCHITECTURES, all_jobs


def _worker(job: dict) -> dict:
    """Runs in a separate process: one torch thread, so N workers use N cores."""
    import torch

    torch.set_num_threads(1)
    from tktd.runner import run_fit

    try:
        summary = run_fit(
            data_path=job["path"],
            mixture_cols=list(job["cols"]),
            run_id=f"{job['ds_id']}_{job['arch']}",
            arch=ARCHITECTURES[job["arch"]],
            n_restarts=job["restarts"],
            adam_steps=job["adam_steps"],
            adam_lr=job["lr"],
            lbfgs_steps=job["lbfgs_steps"],
            seed=job["seed"],
            use_prior=not job["no_prior"],
            laplace=job["laplace"],
            output_dir=job["output_dir"],
            mlflow_experiment=job["mlflow_experiment"],
            verbose=True,
        )
        return {"ok": True, "ds_id": job["ds_id"], "arch": job["arch"],
                "deviance": summary["fit"]["deviance"], "aic": summary["fit"]["aic"],
                "converged": summary["fit"]["converged"]}
    except Exception as exc:
        return {"ok": False, "ds_id": job["ds_id"], "arch": job["arch"],
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", choices=("all", "real", "artificial"), default="all")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--restarts", type=int, default=12)
    p.add_argument("--adam-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--lbfgs-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-prior", action="store_true", help="MLE instead of MAP")
    p.add_argument("--laplace", action="store_true")
    p.add_argument("--output-dir", default="output_torch")
    p.add_argument("--mlflow-experiment", default="TKTD-NeuralODEs-torch-paper")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not Path("data").is_dir():
        print("Run this from the repo root (the folder containing data/, R/, src/).",
              file=sys.stderr)
        return 2

    jobs = [
        {"ds_id": ds_id, "path": path, "cols": cols, "arch": arch.name,
         "restarts": args.restarts, "adam_steps": args.adam_steps, "lr": args.lr,
         "lbfgs_steps": args.lbfgs_steps, "seed": args.seed, "no_prior": args.no_prior,
         "laplace": args.laplace, "output_dir": args.output_dir,
         "mlflow_experiment": args.mlflow_experiment}
        for ds_id, path, cols, arch in all_jobs(args.scope)
    ]

    print(f"Dispatching {len(jobs)} runs across {args.jobs} workers "
          f"(scope={args.scope}, restarts={args.restarts})", flush=True)
    t0 = time.time()

    done, failed = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            (done if res["ok"] else failed).append(res)
            if not res["ok"]:
                print(f"  FAILED {res['ds_id']}/{res['arch']}: {res['error']}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{len(done)}/{len(jobs)} runs finished in {elapsed / 60:.1f} min. "
          f"Results in {args.output_dir}/")

    if done:
        print("\nBest architecture per data set (lowest AIC):")
        by_ds: dict[str, list[dict]] = {}
        for r in done:
            by_ds.setdefault(r["ds_id"], []).append(r)
        for ds_id in sorted(by_ds):
            best = min(by_ds[ds_id], key=lambda r: r["aic"])
            print(f"  {ds_id:<12} {best['arch']:<32} AIC={best['aic']:10.2f}")

    n_unconverged = sum(1 for r in done if not r["converged"])
    if n_unconverged:
        print(f"\n{n_unconverged} run(s) had no second restart agreeing with the best "
              f"optimum. The objective is non-convex, so consider --restarts "
              f"{max(24, args.restarts * 2)} for the final numbers.")

    if failed:
        print(f"\n{len(failed)} run(s) failed:")
        for r in failed:
            print(f"  {r['ds_id']}/{r['arch']}: {r['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
