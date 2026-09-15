# TKTD-IT in PyTorch

A differentiable reimplementation of the TK-(NN)-TD-IT model of
`src/JAGS_TKTD_IT_generic.txt`, fitted by gradient-based optimisation instead of
MCMC. It reproduces the paper's 47 calibration runs, and is the starting point
for adding molecular information (see `CHEMISTRY_EXTENSION.md`).

## Why PyTorch

The JAGS model contains no discrete latent variables, so the log-likelihood is
differentiable end to end. That makes the choice straightforward:

- **Speed.** All 47 fits take a few minutes on a multi-core machine, against
  hours for the MCMC pipeline. The whole loop becomes interactive.
- **The extension needs it.** Predicting `kd`, `alpha` and the interaction
  matrix from molecular descriptors means embedding neural networks inside the
  model. That is natural here and not expressible in JAGS at all.
- **Chemistry lives in Python.** RDKit, and any pretrained molecular model, are
  Python-first. Fighting that from R would cost more than it saves.
- **Nothing is given up.** Uncertainty is still available through the Laplace
  approximation (`--laplace`), a parametric bootstrap, or by handing the same
  model to NumPyro/Pyro for full Bayesian inference later.

CPU only: these models have a few hundred parameters. Parallelism comes from
running many fits at once, one thread each, not from a GPU.

## The port is verified, not just plausible

`validate_against_jags.{R,py}` runs the real JAGS pipeline, takes one MCMC
iteration's parameter values together with the deviance JAGS computed there,
injects those exact values into the PyTorch model, and compares:

```
[PASS] n_exp                     JAGS=302.14131454  torch=302.14131454  |diff|=1.1e-12
[PASS] n                         JAGS=235.76286385  torch=235.76286385  |diff|=2.8e-14
[PASS] nn_ReLU_n_exp             JAGS=249.72261326  torch=249.72261348  |diff|=2.2e-07
[PASS] nn_n_exp                  JAGS=307.77790094  torch=307.77790094  |diff|=5.1e-13
[PASS] nn_ReLU_nn_ReLU_n_exp     JAGS=253.44584570  torch=253.44584570  |diff|=4.0e-13
```

This separates the model from the inference algorithm: agreement at identical
parameter values proves the likelihood is the same function, which comparing two
independent fits could never establish. (The 2e-07 case is a ReLU kink: a
pre-activation sits near zero, where a rounding difference flips the branch.)

Reproduce it with:

```bash
Rscript python/validate_against_jags.R output/validation

source venv/bin/activate
python3 python/validate_against_jags.py output/validation
deactivate
```

## Install

```
MYIP=62.210.78.147
rsync -avz python/ root@$MYIP:~/maxim_TK_NeuralODEs_TD/python/
rsync -avz Dockerfile.torch root@$MYIP:~/maxim_TK_NeuralODEs_TD/
```

```bash
pip install -r python/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

or use the image, which needs no R and no JAGS:

```bash
docker build -f Dockerfile.torch -t tktd-torch:latest .
```

## Run

One fit:

```bash
python3 python/run_tktd_torch.py --list-arch      # the 8 paper architectures

python3 python/run_tktd_torch.py \
    --data data/data_artificial_synergism.csv \
    --arch nn_ReLU_n_exp --id synergism
```

Any architecture the JAGS file can express, via the same flags:

```bash
python3 python/run_tktd_torch.py \
    --data data/MaXim__raw_datasets__set1_CLEAN.csv \
    --hidden-layers 6,6 --out-exp 1 --alpha-split 1 --id set1_custom
```

All 47 paper runs:

```bash
python3 python/reproduce_paper_torch.py --jobs $(nproc)
python3 python/reproduce_paper_torch.py --scope artificial     # 15 runs
python3 python/reproduce_paper_torch.py --restarts 4 --adam-steps 500   # smoke test
```

With MLflow, using the same server as the JAGS pipeline (`172.17.0.1` is the
docker bridge gateway; check yours with `ip addr show docker0`):

```bash
export MLFLOW_TRACKING_URI=http://172.17.0.1:5000
python3 python/reproduce_paper_torch.py --jobs $(nproc)
```

Each run writes `output_torch/torch_<id>_L<n>_M<m>_out<..>_alphasplit<..>_aact<..>.json`
plus a `_predictions.csv`. The tag matches the R engine's naming so JAGS and
PyTorch outputs pair up automatically.

## Compare against the JAGS results

```bash
Rscript python/summarize_jags_rda.R output output/jags_summary.csv
python3 python/compare_jags_torch.py output/jags_summary.csv output_torch
```

### How to read the comparison

**`kd` and `hb` should match closely** — they are well identified, and in
testing they land inside the JAGS 95% credible interval essentially always.

**`alpha` and `beta` will not match individually, and that is not a bug.** The
model has an exact continuous symmetry: scaling the last layer's weights and
biases by `c`, `log(alpha)` by `c`, and `beta` by `1/c` leaves every predicted
survival probability unchanged. The likelihood therefore cannot identify `alpha`
and `beta` separately, only their combination. MCMC explores that ridge and
reports a posterior mean somewhere along it; the MAP sits where the prior pins
it. Different points, identical predictions. The same caveat applies to the
network weights, which are additionally invariant to permuting hidden units.

**Compare the fitted curves instead.** `compare_jags_torch.py` reports
`curve_rmse`, the RMSE between the two pipelines' fitted survivor counts. In
testing this is well under one individual out of 100.

**Deviance: expect the PyTorch value to be lower by roughly `k`.** A posterior
mean deviance always exceeds the deviance at the mode, and the gap is the
effective number of parameters. A gap of about `k` is agreement.

Note that `mean_deviance` and `mean_WAIC`, which `fit_TKTD_bayes()` logs to
MLflow, are averages **per observation** (the dic module monitors every observed
node). Multiply by the number of rows to compare against a total deviance;
`summarize_jags_rda.R` records both.

## Uncertainty (`--laplace`)

`--laplace` adds a Gaussian approximation at the mode. It always reports
**standard errors** from the inverse Hessian, which are trustworthy for the
identified parameters (`kd`, `hb`).

It also tries to compute WAIC and a posterior mean deviance, but **expect this
to be refused on most runs**, with the reason recorded in the JSON. The cause is
the scaling ridge described above: the posterior is far from Gaussian along it,
so sampling from the quadratic approximation wanders into regions of negligible
likelihood and the resulting WAIC would be meaningless. The check compares the
mean deviance of the draws against the deviance at the mode, and refuses when
the gap far exceeds the parameter count.

**Use AIC or BIC for model comparison.** They are computed on every run and need
no posterior. If you specifically need WAIC, the honest route is full Bayesian
inference — either the existing JAGS pipeline, or handing this model to
NumPyro/Pyro.

A useful sanity check in the output: when `alpha_split = 0`, the unused
`alpha_log10` entries come back with a standard error equal to their prior
standard deviation (0.5), because the likelihood never touches them.

## MAP or MLE

The default is MAP: the JAGS priors are kept and act as penalties, so the
optimum is the posterior mode. This is both the estimate most comparable to a
JAGS posterior and the better-behaved objective, since the bridge weights are
only weakly identified by the likelihood alone. `--no-prior` gives plain MLE.

## Convergence

The objective is non-convex, with genuinely distinct local optima; different
starting points reach different ones. The fitter therefore runs `--restarts`
independent optimisations from prior draws (Adam with a decaying learning rate,
then L-BFGS to polish) and keeps the best.

`converged` in the JSON means at least two restarts agreed on the best optimum.
When the batch driver reports runs that did not converge, raise `--restarts`
before trusting the numbers — with 48 cores this is cheap. `restart_objectives`
records every restart so the spread is visible.

Restarts alternate between two initialisation strategies: a raw draw from the
priors, and one where the output bias is shifted so the initial damage lands on
the tolerance threshold. Straight from the priors the bridge output is of order
1 while `alpha` starts at `10^-1.5`, which saturates the survival curve and
leaves almost no gradient signal; the shift fixes that. Each wins on different
architectures, so mixing them widens the search.

## Layout

| file | role |
|---|---|
| `tktd/model.py` | the model, ported line by line from the JAGS file |
| `tktd/bridge.py` | `W_mask`/`b_mask` construction (port of `.build_generic_bridge_data`) |
| `tktd/data.py` | CSV preprocessing (port of `.prepare_TKTD_data`) |
| `tktd/fit.py` | multi-restart MAP/MLE, Laplace approximation |
| `tktd/architectures.py` | the paper's 8 architectures and 7 data sets |
| `tktd/runner.py` | one fit end to end, plus MLflow logging |
| `run_tktd_torch.py` | single-run CLI |
| `reproduce_paper_torch.py` | all 47 runs in parallel |
| `validate_against_jags.{R,py}` | the exactness check above |
| `summarize_jags_rda.R`, `compare_jags_torch.py` | JAGS vs PyTorch comparison |
| `CHEMISTRY_EXTENSION.md` | design for adding logP/Kow/Koa/SMILES and ADME |

## Numerical note

The survival probability is built in log space:

```
F     = D^beta / (D^beta + alpha^beta) = sigmoid(beta * (log D - log alpha))
1 - F = sigmoid(-beta * (log D - log alpha))
max_t exp(z) = exp(max_t z)
```

These identities are exact, not approximations. Computing `D^beta` directly
overflows — `D` can reach `exp(30)` and `beta` can reach 100 — giving `inf/inf`.
The log-space form is what makes the deviance match JAGS to 1e-12 rather than
returning `nan`.
