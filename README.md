# Maxim TK NeuralODEs TD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22705948.svg)](https://doi.org/10.5281/zenodo.22705948)
[![Paper](https://img.shields.io/badge/Published_in-PLOS_Comp_Biol-blue)](https://doi.org/10.1371/journal.pcbi.1013681)
[![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)](#)

This repository contains the code and datasets used to model the effects of chemical mixtures on survival 
using a Bayesian Neural Ordinary Differential Equations (Neural ODEs) framework coupled with 
Toxicokinetic-Toxicodynamic (TKTD) models.

## Citation

If you use this code, the models, or the datasets in your research, please
cite the original article published in *PLOS Computational Biology*:

> **Baudrot, V., Cedergreen, N., Kleiber, T., Gergs, A., & Charles, S. (2025).
A Bayesian neural ordinary differential equations framework to study the effects 
of chemical mixtures on survival.** *PLoS Computational Biology, 21(11), e1013681*.
DOI: [10.1371/journal.pcbi.1013681](https://doi.org/10.1371/journal.pcbi.1013681)


```bibtex
@article{baudrot2025bayesian,
  title={A Bayesian neural ordinary differential equations framework to study the effects of chemical mixtures on survival},
  author={Baudrot, Virgile and Cedergreen, Nina and Kleiber, Thomas and Gergs, Andr{\'e} and Charles, Sandrine},
  journal={PLoS Computational Biology},
  volume={21},
  number={11},
  pages={e1013681},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```


---

## Repository Structure (Current R Implementation)

The current version of the repository provides the exact R scripts used to generate the results and figures presented in the paper.

```text
📦 maxim_TK_NeuralODEs_TD
 ┣ 📂 R/               # Core R scripts for Bayesian TKTD and Neural ODE models
 ┃ ┣ 📜 Bayes_TKNNTD.R # Main script for the Neural ODE TKTD model
 ┃ ┣ 📜 Bayes_TKTD_n.R # Baseline TKTD models
 ┃ ┣ 📜 data_summary.R # Scripts for data exploration and summary
 ┃ ┗ 📜 ...            # Various model architectures (ReLU, exponential, etc.)
 ┣ 📂 data/            # Cleaned experimental and artificial datasets
 ┃ ┣ 📜 MaXim__raw_datasets__set1_CLEAN.csv
 ┃ ┣ 📜 data_artificial_additive.csv
 ┃ ┗ 📜 ...
 ┣ 📂 img/             # Output directory for generated prediction plots
 ┗ 📜 README.md
```

## Install

1) Create R environment with JAGS and the library `Rjags`:

```shell
sudo apt-get update
sudo apt-get -y install r-base
apt-get -y update && apt-get -y install jags
R -e "install.packages('coda',dependencies=TRUE, repos='http://cran.rstudio.com/')"
R -e "install.packages('rjags',dependencies=TRUE, repos='http://cran.rstudio.com/')"
```

2) Copy all require files:

The variable `MIP` is the `IP` of the remote machine.

```shell
MIP="XXX.XXX.XXX.XXX"
ssh root@$MIP

mkdir src
mkdir data
mkdir output
```

Copy JAGS models:

```shell
scp -r src/JAGS_TKTD_IT_n.txt root@$MIP:~/src/
...
```

Copy R scripts:

```shell
scp -r Bayes_TKTD_n.R root@$MIP:~/
...
```

Copy Data:

```shell
scp -r data/data_artificial_synergism.csv root@$MIP:~/data/
...
```

Copy run files:

```shell
scp -r run_*.R root@$mip:~/
```

## Run an experiment

- Use `run_*.R` to reproduce the results.

## License

```
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
```