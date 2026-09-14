# Deploying and running this repository on a Scaleway machine

This guide explains how to build the repository's Docker image (`Dockerfile`
at the repo root) and run it on a Scaleway instance, with a quick-test
example to check that everything works, then how to reproduce every
calibration run of the paper — Baudrot et al., *A Bayesian neural ordinary
differential equations framework to study the effects of chemical mixtures
on survival*, PLOS Computational Biology 2025 — using **only** the generic
bridge (`src/JAGS_TKTD_IT_generic.txt`), dispatched in parallel across a
multi-core machine, with optional MLflow experiment tracking.

> **Status**: the Docker build itself has been confirmed working (thanks to
> your feedback — the `r-cran-rjags` fix below). The `alpha-activation` flag
> and the `N_EXPOSURE`-sized `--hidden-layers` convention described in
> section 7 were added/corrected after that, and the `run/reproduce_paper.sh`
> job matrix (dataset sizes, architecture flags) was validated logically
> (correct `--hidden-layers`/flag values for all 47 runs) but not run
> against real JAGS in this session — I still don't have a working `docker
> build` here (network policy blocks Docker Hub layer downloads). Please
> re-run the quick test in section 6 before launching the full matrix.

## 1. What the Docker image does

`Dockerfile` (repo root):
- starts from `r-base:4.4.2` (official R image, Debian-based);
- installs `jags` plus the prebuilt Debian packages `r-cran-coda`/
  `r-cran-rjags` (used by every `R/Bayes_*.R`) and `r-cran-readr`/
  `r-cran-dplyr` (used by `R/data_summary.R`) — all via `apt-get`, so no
  compilation from CRAN sources and no separate JAGS headers package
  needed (`r-base`'s Debian base doesn't ship one under the
  `libjags-dev` name);
- copies the whole repository into `/repo` (`.dockerignore` excludes
  `.git/` and `img/`, ~180 MB combined, neither needed to run the scripts);
- sets `WORKDIR /repo`, because **every script in this repo uses paths
  relative to the repo root** (`src/JAGS_....txt`, `data/....csv`,
  `output/...`).

No fixed `ENTRYPOINT`: the image works both as an interactive shell
(`bash`) and for launching `Rscript ...` directly.

## 2. Prerequisites

- Docker installed locally (to build/test the image before shipping it).
- A Scaleway account, and if you use the CLI:
  [`scw`](https://github.com/scaleway/scaleway-cli) installed and
  configured (`scw init` with your API keys). Everything below is also
  doable from the [Scaleway Console](https://console.scaleway.com) without
  the CLI.
- Exact instance-type / marketplace-image names change over time in
  Scaleway's catalog: check `scw instance server-type list` /
  `scw marketplace image list` (or the Console) if a command below fails
  because of an outdated name.

## 3. Build the image locally (do this before shipping anything)

```bash
cd maxim_TK_NeuralODEs_TD
docker build -t tktd-neuralodes:latest .
```

This build needs no GPU (~2-5 min; everything is installed as prebuilt
Debian packages, no compilation).

## 4. Provision a Scaleway instance

Since you plan to run many simulations **in parallel**, size the instance
for that: `run/reproduce_paper.sh` (section 7.3) dispatches one `docker
run` per CPU core by default (via GNU parallel), and each run is one
single-threaded R/JAGS process (JAGS runs its 3 MCMC chains sequentially
within one process — parallelism here comes from running many *separate*
fits side by side, not from multi-threading a single fit). So: **more
vCPUs = more concurrent runs**, roughly linearly. A machine with, say, 16
or 32 vCPUs will run the full 47-run paper matrix (section 7.3) 16x/32x
faster than a 1-core box. RAM needs are modest per run (each JAGS model
here is small); a few GB per core is plenty.

**Option A — Web Console**: create an instance (Compute > Instances >
Create instance), pick a "Docker" marketplace image if available (Docker
preinstalled), otherwise a standard Ubuntu LTS image. Pick a
compute-optimized or general-purpose type with as many vCPUs as your
budget allows (e.g. a `POP2`/`PRO2`/`COPARM1` high-core-count type — check
current names in the Console).

**Option B — `scw` CLI** (example with a 16-vCPU type; adjust `type=` to
whatever high-core-count type is current in your region):

```bash
scw instance server create \
  type=PRO2-M \
  zone=fr-par-1 \
  image=ubuntu_jammy \
  name=tktd-neuralodes \
  root-volume=l:40G \
  ip=new
```

Grab the public IP that gets printed, then connect:

```bash
ssh root@<PUBLIC_IP>
```

Install Docker and GNU parallel (needed for section 7.3) if not already
present:

```bash
curl -fsSL https://get.docker.com | sh
apt-get update && apt-get install -y parallel
```

## 5. Ship the code to the instance

The whole repository (including `data/*.csv`, already tracked in git) is a
few MB — three equivalent ways to get it there:

**Option A — `git clone` directly on the instance (simplest)**:

```bash
ssh root@<PUBLIC_IP> "git clone <REPO_URL> maxim_TK_NeuralODEs_TD"
```

**Option B — `rsync`/`scp` from your machine** (useful if you have local
changes that aren't pushed yet):

```bash
rsync -avz --exclude '.git' --exclude 'img' --exclude 'output' \
  ./maxim_TK_NeuralODEs_TD/ root@<PUBLIC_IP>:~/maxim_TK_NeuralODEs_TD/
```

**Option C — go through the Scaleway Container Registry** (avoids
rebuilding the image on every instance; useful if you spin up several):

```bash
# locally
scw registry namespace create name=tktd-neuralodes region=fr-par
docker login rg.fr-par.scw.cloud -u nologin -p <SCW_SECRET_KEY>
docker tag tktd-neuralodes:latest rg.fr-par.scw.cloud/tktd-neuralodes/tktd-neuralodes:latest
docker push rg.fr-par.scw.cloud/tktd-neuralodes/tktd-neuralodes:latest

# on the instance
docker login rg.fr-par.scw.cloud -u nologin -p <SCW_SECRET_KEY>
docker pull rg.fr-par.scw.cloud/tktd-neuralodes/tktd-neuralodes:latest
docker tag rg.fr-par.scw.cloud/tktd-neuralodes/tktd-neuralodes:latest tktd-neuralodes:latest
```

With options A/B, build the image on the instance (same as step 3):

```bash
cd ~/maxim_TK_NeuralODEs_TD
docker build -t tktd-neuralodes:latest .
```

## 6. Quick test (do this first, ~1 minute)

On the instance (or locally, before shipping anything to Scaleway at all).
Every example in this guide uses **only** the generic bridge
(`--bridge generic`, i.e. `src/JAGS_TKTD_IT_generic.txt`) — it fully
replaces the 11 legacy per-architecture JAGS files (kept in `src/` only for
internal cross-checking, not used below):

```bash
mkdir -p output
docker run --rm -v "$PWD/output:/repo/output" tktd-neuralodes:latest \
  Rscript run/run_TKTD_bayes.R \
    --data data/data_artificial_additive.csv \
    --bridge generic --out-exp 1 \
    --n-update 200 --n-iter 200 --n-iter-waic 100
```

This uses the smallest dataset in the repo (150 rows, 2 compounds) and the
simplest bridge configuration (no hidden layer, `--out-exp 1` — this is the
paper's `n_exp` model, see the table in section 7.2), with a deliberately
small number of MCMC iterations. It does not produce usable inference,
only a check that the R + JAGS + rjags + repo stack works. Should finish in
well under a minute.

**Expected output** (in the terminal):

```
Auto-detected exposure columns for data_artificial_additive.csv: A, B
[1] "START MODEL additive JAGS_TKTD_IT_generic_L1_M2_out1_alphasplit0_aact1"
[1] "START UPDATE additive JAGS_TKTD_IT_generic_L1_M2_out1_alphasplit0_aact1"
[1] "START TRACE additive JAGS_TKTD_IT_generic_L1_M2_out1_alphasplit0_aact1"
[1] "START WAIC additive JAGS_TKTD_IT_generic_L1_M2_out1_alphasplit0_aact1"
[1] "SAVED output/mcmc_additive_JAGS_TKTD_IT_generic_L1_M2_out1_alphasplit0_aact1.rda"
```

**Check**:

```bash
ls -la output/
# should contain a file named mcmc_additive_JAGS_TKTD_IT_generic_L1_...rda
```

If that file exists, the setup works.

## 7. Reproducing the paper — generic bridge only

### 7.1 One run, by hand

```bash
docker run --rm -v "$PWD/output:/repo/output" tktd-neuralodes:latest \
  Rscript run/run_TKTD_bayes.R \
    --data data/MaXim__raw_datasets__set1_CLEAN.csv \
    --bridge generic --hidden-layers 6 --out-exp 1 \
    --id 1_nn_ReLU_n_exp
```

(Columns are auto-detected here; pass `--mixture col1,col2,...` to force
them, e.g. when using `R/Bayes_TK-NN-TD_generic.R` directly instead of the
`run/run_TKTD_bayes.R` wrapper.) `--hidden-layers 6` matches this
dataset's compound count (set1 has 6 compounds) — **this is not
arbitrary**: per Table 1 of the paper, every hidden layer's width equals
the number of compounds `n` in the fitted dataset (a `n × n` weight
matrix), never a fixed value. `run/reproduce_paper.sh` below computes this
automatically per dataset.

### 7.2 The 8 architectures of the paper, as generic-bridge flags

Table 1 (calibration) defines 6 named bridges; Table 3 adds 2 more
"split" variants (alpha estimated per compound, through the **same**
network as the damage bridge). All 8 are reachable through `--bridge
generic` with `N` = number of compounds in the dataset being fit:

| Paper label                     | `--hidden-layers`  | `--neg-slope` | `--out-exp` | `--alpha-split` | `--alpha-out-exp` | `--alpha-activation` |
|----------------------------------|---------------------|---------------|-------------|------------------|--------------------|------------------------|
| `n`                              | *(omit)*            | —             | `0`         | `0`              | —                  | —                      |
| `n_exp`                          | *(omit)*            | —             | `1`         | `0`              | —                  | —                      |
| `n_exp_split`                    | *(omit)*            | —             | `1`         | `1`              | `0`                | —                      |
| `nn_n_exp`                       | `N`                 | `1`           | `1`         | `0`              | —                  | —                      |
| `nn_ReLU_n_exp`                  | `N`                 | `0`           | `1`         | `0`              | —                  | —                      |
| `nn_ReLU_n_exp_split`            | `N`                 | `0`           | `1`         | `1`              | `1`                | `0`                    |
| `nn_ReLU_nn_ReLU_n_exp`          | `N,N`               | `0`           | `1`         | `0`              | —                  | —                      |
| `nn_ReLU_nn_ReLU_nn_ReLU_n_exp`  | `N,N,N`             | `0`           | `1`         | `0`              | —                  | —                      |

Notes:
- `--neg-slope 0` = ReLU, `1` = identity (no activation) — `nn_n_exp` is
  explicitly the *no-activation* perceptron per Table 1 ("without
  activation function"), unlike every other `nn_*` bridge.
- `--alpha-activation 0` is what makes `nn_ReLU_n_exp_split` an exact
  reproduction of the original `Bayes_TKNNTD_split.R`: that script applies
  ReLU to the damage (`D`) branch but **not** to the alpha branch, even
  though both share the same weights. This is a real asymmetry in the
  original code, not a simplification on my part — I verified it by
  re-reading `src/JAGS_TKNNTD_split_IT.txt` line by line. Every other
  architecture above has `alpha_split = 0`, so `--alpha-activation` doesn't
  affect them (the alpha branch is unused).
- Per the paper (Methods 2.4 and the Fig 3 caption): the **4 real MaXim
  datasets** (set1–set4) were each calibrated with **all 8** architectures
  above; the **3 artificial datasets** (additive/antagonism/synergism) were
  each calibrated with only **5** of them — `n`, `n_exp`, `nn_ReLU_n_exp`,
  `nn_ReLU_nn_ReLU_n_exp`, `nn_ReLU_nn_ReLU_nn_ReLU_n_exp` (no splitting).
  That's 4×8 + 3×5 = **47 runs** total, exactly what `reproduce_paper.sh`
  dispatches below.
- MCMC settings match the paper's (Methods 2.4): 3 chains, 5000 warm-up
  iterations (`--n-update`), 2500 kept for inference (`--n-iter`) — these
  are already `run_TKTD_bayes.R`'s defaults, no need to pass them.

### 7.3 All 47 runs, in parallel: `run/reproduce_paper.sh`

```bash
chmod +x run/reproduce_paper.sh   # once, if the +x bit didn't survive transfer
JOBS=16 ./run/reproduce_paper.sh
```

This script (at the repo root, see its header comment for full details):
computes `N` and the right `--hidden-layers`/flags per the table above for
every (dataset, architecture) combination, and dispatches all 47 `docker
run` calls through GNU parallel, bounded by `JOBS` (default:
`$(nproc)` — one job per core). Useful env vars:

```bash
SCOPE=real ./run/reproduce_paper.sh          # only the 4 MaXim sets (32 runs)
SCOPE=artificial ./run/reproduce_paper.sh    # only the 3 artificial sets (15 runs)
N_UPDATE=200 N_ITER=200 N_ITER_WAIC=100 ./run/reproduce_paper.sh   # a fast dry run of all 47, to sanity-check the whole matrix before committing to the full-length one
JOBS=4 IMAGE=tktd-neuralodes:latest ./run/reproduce_paper.sh
```

Run the `N_UPDATE=200 ...` fast pass first — it exercises every
architecture/dataset combination (so any misconfigured flag surfaces
immediately) without waiting through 47 full-length MCMC fits.

### 7.4 `R/data_summary.R` (descriptive stats for the 4 MaXim datasets)

```bash
docker run --rm tktd-neuralodes:latest Rscript R/data_summary.R
```

## 8. Tracking runs with MLflow

Running 47 (or more, if you explore beyond the paper's grid) parallel fits
produces 47 separate log streams and `.rda` files — MLflow gives you one
dashboard instead. Support is already built into `fit_TKTD_bayes()`
(`R/Bayes_TK-NN-TD_generic.R`): every run logs its parameters (bridge
config, dataset, MCMC settings) and two summary metrics (`mean_deviance`,
`mean_WAIC`, averaged from the WAIC trace) plus the `.rda` file as an
artifact — **automatically, whenever `MLFLOW_TRACKING_URI` is set in the
environment**. No code changes needed; it degrades to a no-op (with a
one-line warning) if the `mlflow` R package isn't installed or the server
is unreachable, so it never breaks a fit — meaning this whole section is
opt-in and skippable.

**Image**: the R `mlflow` package has no prebuilt Debian package (unlike
coda/rjags in the main `Dockerfile`), so it lives in a separate,
optional `Dockerfile.mlflow` layered on top of the core image — this keeps
the (now working) core image isolated from a CRAN compile step I haven't
been able to test in this session:

```bash
docker build -t tktd-neuralodes:latest .              # core image, as before
docker build -f Dockerfile.mlflow -t tktd-neuralodes:mlflow .   # + R mlflow package
```

Use `tktd-neuralodes:mlflow` (via `IMAGE=tktd-neuralodes:mlflow`) instead
of `:latest` for any run you want tracked; untracked runs can keep using
`:latest` as normal.

**Setup (once per instance)** — run ONE MLflow tracking server on the
Scaleway host itself (not inside the analysis containers), backed by
SQLite, so every parallel container can log to it concurrently:

```bash
apt-get install -y pipx
pipx install mlflow
mkdir -p ~/mlflow-data
/root/.local/bin/mlflow server \
  --backend-store-uri sqlite:///~/mlflow-data/mlflow.db \
  --default-artifact-root ~/mlflow-data/artifacts \
  --host 0.0.0.0 --port 5000 &
```

(`pipx` rather than a plain `pip install`: recent Ubuntu/Debian images refuse
system-wide `pip install` by default — PEP 668, "externally-managed-
environment" — to protect the OS's own Python packages. `pipx` installs
`mlflow` into its own isolated environment instead, which is what pip's own
error message recommends. The full `/root/.local/bin/mlflow` path avoids
needing to reopen the shell for `pipx`'s `PATH` update to take effect; once
you've reconnected via SSH at least once since installing, plain `mlflow`
also works.)

**Point every `docker run` at it** by exporting `MLFLOW_TRACKING_URI`
before calling `reproduce_paper.sh` — it forwards the variable into each
container automatically if set (see its `run_one()`), which already passes
`--add-host=host.docker.internal:host-gateway` on every `docker run` so
containers can reach the host (plain Docker Engine on Linux, which is what
a Scaleway instance runs, does **not** resolve `host.docker.internal` by
default the way Docker Desktop does — this flag is what makes it work
here; needs Docker >= 20.10):

```bash
export MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
IMAGE=tktd-neuralodes:mlflow JOBS=16 ./run/reproduce_paper.sh
```

If you write your own `docker run` command instead of using the script,
add that same `--add-host` flag, or use the Docker bridge gateway IP
directly (`ip addr show docker0`, typically `172.17.0.1`):
`export MLFLOW_TRACKING_URI="http://172.17.0.1:5000"`.

**View the dashboard**: either open `http://<PUBLIC_IP>:5000` directly (if
your Scaleway security group allows inbound port 5000), or tunnel it
without opening the port:

```bash
ssh -N -L 5000:localhost:5000 root@<PUBLIC_IP>
# then open http://localhost:5000 in your local browser
```

If you'd rather not run a tracking server at all: skip this section
entirely, everything still works — the `.rda` filenames already encode the
full architecture (see the `out_tag` pattern in the quick-test output
above), which is enough to tell runs apart on disk.

## 9. Retrieving results

`.rda` files (MCMC traces) land in `output/` on the instance (the mounted
volume). To pull them back:

```bash
scp -r root@<PUBLIC_IP>:~/maxim_TK_NeuralODEs_TD/output ./output
# or
rsync -avz root@<PUBLIC_IP>:~/maxim_TK_NeuralODEs_TD/output/ ./output/
```

## 10. Notes / troubleshooting

- **Full-run duration**: heavily depends on the dataset (150 to ~1500
  rows) and the architecture (linear bridge vs. 3-hidden-layer network);
  with the paper's MCMC settings, expect anywhere from a few minutes to
  several tens of minutes per run — always validate with the quick test
  (section 6) and the `N_UPDATE=200 ...` fast pass (section 7.3) before
  committing to the full 47-run matrix.
- **Don't forget the `-v .../output` mount**: `run/reproduce_paper.sh`
  and the examples above already include it; if you write your own
  `docker run` command, without it `--rm` deletes the container (and the
  `.rda` files) at the end of the run.
- **Machine idle between sessions**: remember to stop/delete the Scaleway
  instance when you're done (`scw instance server stop/delete` or via the
  Console) to avoid unnecessary billing.
- **CPU only**: none of these scripts use a GPU; no need to pick a
  Scaleway GPU instance (`RENDER-S`/`H100-*`) for this repo.
