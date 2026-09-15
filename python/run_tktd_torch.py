#!/usr/bin/env python3
"""Single TKTD-IT fit from the command line: the PyTorch counterpart of
run/run_TKTD_bayes.R.

Two ways to pick a model:

  --arch NAME     one of the paper's 8 architectures (see --list-arch); the
                  hidden-layer widths are resolved to the data set's compound
                  count automatically, as in Table 1's 'n x n' matrices.

  explicit flags  --hidden-layers/--neg-slope/--out-exp/--alpha-split/
                  --alpha-out-exp/--alpha-activation, matching the JAGS data
                  flags one for one.

Examples (from the repo root):

    python3 python/run_tktd_torch.py --data data/data_artificial_additive.csv \
        --arch nn_ReLU_n_exp --id additive

    python3 python/run_tktd_torch.py \
        --data data/MaXim__raw_datasets__set1_CLEAN.csv \
        --hidden-layers 6,6 --out-exp 1 --alpha-split 1 --id set1_custom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tktd.architectures import ARCHITECTURES
from tktd.runner import run_fit, set_thread_budget


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="path to a CSV in data/")
    p.add_argument("--mixture", help="comma-separated exposure columns (default: auto-detect)")
    p.add_argument("--id", dest="run_id", help="run identifier used in output names")
    p.add_argument("--arch", choices=sorted(ARCHITECTURES), help="a named paper architecture")
    p.add_argument("--list-arch", action="store_true", help="list the named architectures and exit")

    g = p.add_argument_group("explicit architecture (ignored when --arch is given)")
    g.add_argument("--hidden-layers", default="", help="comma-separated widths, e.g. 6,6")
    g.add_argument("--neg-slope", type=float, default=0.0,
                   help="0 = ReLU, 1 = identity, 0.01 = leaky ReLU")
    g.add_argument("--out-exp", type=int, default=1, choices=(0, 1))
    g.add_argument("--alpha-split", type=int, default=0, choices=(0, 1))
    g.add_argument("--alpha-out-exp", type=int, default=1, choices=(0, 1))
    g.add_argument("--alpha-activation", type=int, default=1, choices=(0, 1))
    g.add_argument("--prec-w", type=float, default=4.0, help="weight prior precision")

    o = p.add_argument_group("optimisation")
    o.add_argument("--restarts", type=int, default=12)
    o.add_argument("--adam-steps", type=int, default=3000)
    o.add_argument("--lr", type=float, default=0.2)
    o.add_argument("--lbfgs-steps", type=int, default=500)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--no-prior", action="store_true",
                   help="plain MLE instead of MAP (drops the JAGS priors)")
    o.add_argument("--laplace", action="store_true",
                   help="add a Laplace posterior: standard errors, and WAIC when valid")
    o.add_argument("--laplace-samples", type=int, default=2000)
    o.add_argument("--threads", type=int, default=0, help="0 = let torch decide")

    p.add_argument("--output-dir", default="output_torch")
    p.add_argument("--mlflow-experiment", default="TKTD-NeuralODEs-torch")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_arch:
        print(f"{'name':<32} {'hidden':<8} {'neg_slope':<10} {'out_exp':<8} "
              f"{'a_split':<8} {'a_out_exp':<10} {'a_activation'}")
        for a in ARCHITECTURES.values():
            print(f"{a.name:<32} {a.n_hidden:<8} {a.neg_slope:<10} {a.out_exp:<8} "
                  f"{a.alpha_split:<8} {a.alpha_out_exp:<10} {a.alpha_activation}")
        print("\nEach hidden layer has width n_X (the data set's compound count).")
        return 0

    if not args.data:
        print("--data is required (or use --list-arch)", file=sys.stderr)
        return 2

    if args.threads:
        set_thread_budget(args.threads)

    mixture = [c.strip() for c in args.mixture.split(",")] if args.mixture else None
    hidden = tuple(int(x) for x in args.hidden_layers.split(",") if x.strip())

    run_fit(
        data_path=args.data,
        mixture_cols=mixture,
        run_id=args.run_id,
        arch=ARCHITECTURES[args.arch] if args.arch else None,
        hidden_layers=hidden,
        neg_slope=args.neg_slope,
        out_exp=args.out_exp,
        alpha_split=args.alpha_split,
        alpha_out_exp=args.alpha_out_exp,
        alpha_activation=args.alpha_activation,
        prec_w=args.prec_w,
        n_restarts=args.restarts,
        adam_steps=args.adam_steps,
        adam_lr=args.lr,
        lbfgs_steps=args.lbfgs_steps,
        seed=args.seed,
        use_prior=not args.no_prior,
        laplace=args.laplace,
        laplace_samples=args.laplace_samples,
        output_dir=args.output_dir,
        mlflow_experiment=args.mlflow_experiment,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
