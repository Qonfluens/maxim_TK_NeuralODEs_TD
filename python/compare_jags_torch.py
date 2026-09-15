#!/usr/bin/env python3
"""Compare the JAGS and PyTorch pipelines run by run.

Reads the CSV written by summarize_jags_rda.R and the JSON files written by the
PyTorch runs, pairs them on (run id, architecture tag), and reports:

  1. per-parameter agreement for kd / alpha / beta / hb, including whether the
     PyTorch point estimate falls inside the JAGS 95% credible interval;
  2. fit quality, comparing JAGS' posterior mean deviance against the PyTorch
     deviance at the mode;
  3. whether both pipelines rank the architectures the same way for each data
     set, which is the conclusion the paper actually draws.

On (2), expect the PyTorch deviance to be LOWER, and by roughly the number of
parameters: a posterior mean deviance always exceeds the deviance at the mode
(that gap is precisely the effective-parameter penalty that DIC and WAIC are
built from). A gap of about k is agreement, not disagreement. The ranking
comparison in (3) is the robust check.

Usage (from the repo root):
    Rscript python/summarize_jags_rda.R output output/jags_summary.csv
    python3 python/compare_jags_torch.py output/jags_summary.csv output_torch
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

TAG_RE = re.compile(r"L(\d+)_M(\d+)_out(\d+)_alphasplit(\d+)(?:_aact(\d+))?$")


def normalise_tag(tag: str | float) -> str | None:
    """'JAGS_TKTD_IT_generic_L2_M6_out1_alphasplit0_aact1' -> 'L2_M6_out1_alphasplit0_aact1'."""
    if not isinstance(tag, str):
        return None
    m = TAG_RE.search(tag)
    if not m:
        return None
    L, M, out, split, aact = m.groups()
    return f"L{L}_M{M}_out{out}_alphasplit{split}_aact{aact or '1'}"


def load_torch_runs(torch_dir: Path) -> dict[tuple[str, str], dict]:
    runs = {}
    for path in sorted(torch_dir.glob("torch_*.json")):
        data = json.loads(path.read_text())
        tag = normalise_tag(path.stem)
        if tag is None:
            continue
        runs[(data["run_id"], tag)] = data
    return runs


def torch_param_value(run: dict, name: str) -> float | None:
    """Map a JAGS monitor name ('kd[2]', 'alpha', ...) onto the torch report."""
    params = run["parameters"]
    if name == "alpha":
        return params["alpha"]
    if name == "beta":
        return params["beta"]
    if name == "hb":
        return params["hb"]
    m = re.match(r"^kd\[(\d+)\]$", name)
    if m:
        idx = int(m.group(1)) - 1
        kd = list(params["kd"].values())
        return kd[idx] if 0 <= idx < len(kd) else None
    return None


def curve_rmse(jags_dir: Path, rda_file: str, torch_dir: Path, run: dict) -> float | None:
    """RMSE between the two pipelines' fitted survivor counts, row by row.

    This is the comparison that matters: alpha and beta are individually
    unidentifiable (scaling the last layer by c, log alpha by c and beta by 1/c
    leaves every prediction unchanged), so they can disagree while the models
    agree completely. The fitted curve has no such freedom.
    """
    stem = Path(rda_file).stem
    jags_pred = jags_dir / f"jagspred_{stem.replace('mcmc_', '', 1)}.csv"
    if not jags_pred.exists():
        return None

    torch_pred = None
    for candidate in torch_dir.glob("torch_*_predictions.csv"):
        if candidate.stem.startswith(f"torch_{run['run_id']}_"):
            torch_pred = candidate
            break
    if torch_pred is None:
        return None

    j = pd.read_csv(jags_pred).sort_values("row")
    t = pd.read_csv(torch_pred)
    if len(j) != len(t):
        return None
    diff = j["Nsurv_ppc_mean"].to_numpy() - t["Nsurv_expected"].to_numpy()
    return float((diff ** 2).mean() ** 0.5)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    jags_csv = Path(sys.argv[1])
    torch_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "output_torch")

    if not jags_csv.exists():
        print(f"JAGS summary not found: {jags_csv}\n"
              f"Run: Rscript python/summarize_jags_rda.R output {jags_csv}", file=sys.stderr)
        return 2
    if not torch_dir.is_dir():
        print(f"PyTorch output dir not found: {torch_dir}", file=sys.stderr)
        return 2

    jags = pd.read_csv(jags_csv)
    jags["tag_norm"] = jags["tag"].map(normalise_tag)
    torch_runs = load_torch_runs(torch_dir)

    if not torch_runs:
        print(f"No torch_*.json files in {torch_dir}", file=sys.stderr)
        return 2

    param_rows, fit_rows = [], []
    unmatched = set()

    for (id_set, tag), group in jags.groupby(["id_set", "tag_norm"], dropna=True):
        run = torch_runs.get((id_set, tag))
        if run is None:
            unmatched.add((id_set, tag))
            continue

        metrics = dict(zip(group["parameter"], group["mean"]))
        if "total_deviance" in metrics:
            jags_dev = metrics["total_deviance"]
            fit_rows.append({
                "id_set": id_set, "architecture": run["architecture"],
                "jags_total_deviance": jags_dev,
                "jags_total_WAIC": metrics.get("total_WAIC"),
                "torch_deviance": run["fit"]["deviance"],
                "gap": jags_dev - run["fit"]["deviance"],
                "torch_n_params": run["fit"]["n_params"],
                "torch_aic": run["fit"]["aic"],
                "torch_converged": run["fit"]["converged"],
                "curve_rmse": curve_rmse(jags_csv.parent, group["file"].iloc[0], torch_dir, run),
            })

        for _, row in group.iterrows():
            name = row["parameter"]
            if name in ("mean_deviance", "mean_WAIC", "total_deviance",
                        "total_WAIC", "n_data"):
                continue

            tv = torch_param_value(run, name)
            if tv is None:
                continue
            inside = bool(row["q2.5"] <= tv <= row["q97.5"]) if pd.notna(row["q2.5"]) else None
            rel = abs(tv - row["mean"]) / max(abs(row["mean"]), 1e-12)
            param_rows.append({
                "id_set": id_set, "architecture": run["architecture"], "parameter": name,
                "jags_mean": row["mean"], "jags_q2.5": row["q2.5"], "jags_q97.5": row["q97.5"],
                "torch": tv, "rel_diff": rel, "in_CI": inside,
            })

    if not param_rows and not fit_rows:
        print("No JAGS run could be paired with a PyTorch run.")
        print("Check that both pipelines used the same --id values "
              "(the paper drivers use '<dataset>_<architecture>').")
        return 1

    params_df = pd.DataFrame(param_rows)
    fit_df = pd.DataFrame(fit_rows)

    print("=" * 78)
    print("1. PARAMETER AGREEMENT")
    print("=" * 78)
    if params_df.empty:
        print("  (no parameters paired)")
    else:
        n_in = int(params_df["in_CI"].sum())
        n_tot = int(params_df["in_CI"].notna().sum())
        print(f"  PyTorch estimates inside the JAGS 95% CI: {n_in}/{n_tot} "
              f"({100 * n_in / max(n_tot, 1):.0f}%)")
        print(f"  median relative difference vs JAGS posterior mean: "
              f"{params_df['rel_diff'].median():.3f}")
        print("\n  by parameter:")
        summary = params_df.groupby(params_df["parameter"].str.replace(r"\[\d+\]", "[.]", regex=True)).agg(
            n=("rel_diff", "size"), median_rel_diff=("rel_diff", "median"),
            frac_in_CI=("in_CI", "mean"))
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    print()
    print("=" * 78)
    print("2. FIT QUALITY  (gap = JAGS posterior-mean deviance - torch deviance at mode)")
    print("   A gap of roughly +k, the parameter count, is the expected agreement.")
    print("   curve_rmse = RMSE between the two fitted survivor counts, in individuals.")
    print("=" * 78)
    if fit_df.empty:
        print("  (no deviance paired)")
    else:
        show = fit_df.sort_values(["id_set", "architecture"])
        print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        if fit_df["curve_rmse"].notna().any():
            print(f"\n  median curve RMSE across runs: "
                  f"{fit_df['curve_rmse'].median():.3f} individuals")

    print()
    print("=" * 78)
    print("3. ARCHITECTURE RANKING PER DATA SET")
    print("=" * 78)
    if fit_df.empty:
        print("  (nothing to rank)")
    else:
        # the paper selects on WAIC; fall back to deviance if it was not stored
        use_waic = fit_df["jags_total_WAIC"].notna().all()
        jags_col = "jags_total_WAIC" if use_waic else "jags_total_deviance"
        label = "WAIC" if use_waic else "deviance"
        for id_set, grp in fit_df.groupby(fit_df["id_set"].str.split("_").str[0]):
            if len(grp) < 2:
                print(f"  {id_set:<12} only {len(grp)} architecture(s) fitted, nothing to rank")
                continue
            j_best = grp.loc[grp[jags_col].idxmin(), "architecture"]
            t_best = grp.loc[grp["torch_aic"].idxmin(), "architecture"]
            flag = "same" if j_best == t_best else "DIFFERENT"
            print(f"  {id_set:<12} JAGS(min {label})={j_best:<32} "
                  f"torch(min AIC)={t_best:<32} [{flag}]")

    if unmatched:
        print(f"\n{len(unmatched)} JAGS run(s) had no PyTorch counterpart, e.g. "
              f"{sorted(unmatched)[:3]}")

    out = torch_dir / "comparison_jags_vs_torch.csv"
    pd.concat([params_df.assign(kind="parameter"), fit_df.assign(kind="fit")],
              ignore_index=True).to_csv(out, index=False)
    print(f"\nFull comparison written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
