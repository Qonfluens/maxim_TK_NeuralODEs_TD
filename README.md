# Bayesian Neural ODE TKTD models for chemical mixtures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22705948.svg)](https://doi.org/10.5281/zenodo.22705948)
[![Paper](https://img.shields.io/badge/PLOS_Comp_Biol-10.1371%2Fjournal.pcbi.1013681-blue)](https://doi.org/10.1371/journal.pcbi.1013681)

Code and data accompanying Baudrot et al. (2025), *A Bayesian neural ordinary
differential equations framework to study the effects of chemical mixtures on
survival*, PLOS Computational Biology.

The model couples a toxicokinetic-toxicodynamic (TKTD) description of survival
with a small neural network. Each compound of a mixture has its own
toxicokinetics, giving an internal concentration over time; a neural "bridge"
combines these internal concentrations into a single damage variable; survival
then follows the individual tolerance (IT) death mechanism. Interactions between
compounds (synergy, antagonism) are learnt by the bridge rather than imposed by
an additivity assumption. Parameters are estimated by Bayesian inference in JAGS.

## Contents

| Path | Description |
|---|---|
| `R/Bayes_TK-NN-TD_generic.R` | Generic fitting engine, `fit_TKTD_bayes()`. Covers every architecture in the paper. |
| `R/Bayes_TK*.R` | Original per-architecture scripts used for the paper, kept for cross-checking. |
| `src/JAGS_TKTD_IT_generic.txt` | Parametric JAGS model (free network depth and width). |
| `src/JAGS_*.txt` | Per-architecture JAGS models matching the scripts in `R/`. |
| `run/run_TKTD_bayes.R` | Command-line entry point for a single fit. |
| `run/reproduce_paper.sh` | Runs the 47 calibrations of the paper in parallel. |
| `run/run_all*.R` | Original batch scripts. |
| `data/` | Experimental and artificial data sets (see below). |
| `python/` | PyTorch reimplementation of the same model (see [`python/README.md`](python/README.md)). |
| `Dockerfile`, `Dockerfile.torch`, `Dockerfile.mlflow` | Images for the JAGS pipeline, the PyTorch pipeline and an MLflow server. |

## Data

| File | Content |
|---|---|
| `MaXim__raw_datasets__set1_CLEAN.csv` | Set 1: six fungicides (spiroxamine, prothioconazole, tebuconazole, trifloxystrobin, bixafen, fluopyram) |
| `MaXim__raw_datasets__set2_CLEAN.csv` | Set 2: five insecticides (spiromesifen, deltamethrin, triazophos, tralomethrin, flupyradifurone) |
| `MaXim__raw_datasets__set3_CLEAN.csv` | Set 3: six insecticides (thiacloprid, imidacloprid, cyfluthrin, clothianidin, beta-cyfluthrin, thiodicarb) |
| `MaXim__raw_datasets__set4_CLEAN.csv` | Set 4: five herbicides (flufenacet, diflufenican, metribuzin, flurtamone, aclonifen) |
| `data_artificial_{additive,antagonism,synergism}.csv` | Simulated binary mixtures with a known interaction |

Each file is in long format: one row per replicate and observation time, with
the number of survivors (`Nsurv`) and one column per compound giving its
exposure concentration.

## Model architectures

The bridge between internal concentrations and damage is selected with
`--bridge generic` and the flags below. Hidden layers have width *n*, the number
of compounds in the data set.

| Name | Hidden layers | Activation | Exponential output | Split alpha |
|---|---|---|---|---|
| `n` | 0 | – | no | no |
| `n_exp` | 0 | – | yes | no |
| `n_exp_split` | 0 | – | yes | yes |
| `nn_n_exp` | 1 | linear | yes | no |
| `nn_ReLU_n_exp` | 1 | ReLU | yes | no |
| `nn_ReLU_n_exp_split` | 1 | ReLU | yes | yes |
| `nn_ReLU_nn_ReLU_n_exp` | 2 | ReLU | yes | no |
| `nn_ReLU_nn_ReLU_nn_ReLU_n_exp` | 3 | ReLU | yes | no |

The exact flag values for each architecture are listed in
`run/reproduce_paper.sh`.

## Installation

### Docker (recommended)

The image contains R, JAGS and the required R packages:

```bash
docker build -t tktd-neuralodes:latest .
```

### Local installation

On Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y r-base jags r-cran-coda r-cran-rjags
```

Installing `rjags` from the Debian packages avoids compiling it against the JAGS
headers. All scripts use paths relative to the repository root and must be run
from there.

## Usage

### Single fit

```bash
Rscript run/run_TKTD_bayes.R --data data/data_artificial_synergism.csv
```

Exposure columns are detected automatically; pass `--mixture col1,col2,...` to
set them explicitly. A two-hidden-layer network with a short MCMC run:

```bash
Rscript run/run_TKTD_bayes.R \
    --data data/data_artificial_synergism.csv \
    --bridge generic --hidden-layers 2,2 \
    --n-update 1000 --n-iter 1000 --n-iter-waic 500 \
    --id synergism_test
```

With Docker, mount `output/` so that results persist:

```bash
docker run --rm -v "$PWD/output:/repo/output" tktd-neuralodes:latest \
    Rscript run/run_TKTD_bayes.R --data data/data_artificial_additive.csv
```

Results are saved as `.rda` files in `output/`.

### Reproducing the paper

`run/reproduce_paper.sh` runs all 47 calibrations: the eight architectures on
each of the four experimental sets, and five architectures on each of the three
artificial sets. It requires Docker and GNU parallel and uses one job per core by
default.

```bash
./run/reproduce_paper.sh                                   # all 47 runs
SCOPE=artificial ./run/reproduce_paper.sh                  # artificial sets only
N_UPDATE=200 N_ITER=200 N_ITER_WAIC=100 ./run/reproduce_paper.sh   # quick check
```

Runs are logged to MLflow when `MLFLOW_TRACKING_URI` is set. Deployment on a
remote multi-core machine is described in
[`DEPLOY_SCALEWAY.md`](DEPLOY_SCALEWAY.md).

### PyTorch implementation

`python/` contains a differentiable port of the generic JAGS model, fitted by
maximum a posteriori estimation instead of MCMC. Its likelihood matches JAGS to
numerical precision at identical parameter values, and the full set of 47 runs
completes in minutes. See [`python/README.md`](python/README.md) for
installation, validation against JAGS and comparison of the two pipelines.

## Citation

If you use this code or the data, please cite:

> Baudrot, V., Cedergreen, N., Kleiber, T., Gergs, A., & Charles, S. (2025).
> A Bayesian neural ordinary differential equations framework to study the
> effects of chemical mixtures on survival. *PLoS Computational Biology*,
> 21(11), e1013681. https://doi.org/10.1371/journal.pcbi.1013681

```bibtex
@article{baudrot2025bayesian,
  title     = {A Bayesian neural ordinary differential equations framework to study the effects of chemical mixtures on survival},
  author    = {Baudrot, Virgile and Cedergreen, Nina and Kleiber, Thomas and Gergs, Andr{\'e} and Charles, Sandrine},
  journal   = {PLoS Computational Biology},
  volume    = {21},
  number    = {11},
  pages     = {e1013681},
  year      = {2025},
  publisher = {Public Library of Science San Francisco, CA USA}
}
```

## License

MIT License

Copyright (c) 2024 Qonfluens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
