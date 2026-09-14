# Reproducible image for maxim_TK_NeuralODEs_TD: R + JAGS + rjags/coda,
# with the repo copied in as-is. Same stack as documented in README.md
# ("## Install"), packaged as a reproducible image that can be built once
# and deployed to any machine (local or Scaleway) -- see DEPLOY_SCALEWAY.md.
FROM r-base:4.4.2

# JAGS (the MCMC engine driven via rjags in every R/Bayes_*.R script) +
# the R packages used across the repo's scripts, all as prebuilt Debian
# packages (r-cran-coda/r-cran-rjags for the MCMC fit; r-cran-readr/
# r-cran-dplyr only for R/data_summary.R). Installing rjags this way avoids
# compiling it from CRAN sources, which would need a "libjags-dev"-style
# headers package -- r-base's Debian testing/sid base doesn't ship one
# under that name (only the "jags" runtime package), so `install.packages
# ("rjags")` fails to build there. The r-cran-* packages are prebuilt
# against the matching "jags" package, so no headers/build tools needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        jags \
        r-cran-coda \
        r-cran-rjags \
        r-cran-readr \
        r-cran-dplyr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# Copy the repository as-is: R/, src/, data/, run/ -- every script in this
# repo uses paths RELATIVE TO THE REPO ROOT (e.g. "src/JAGS_....txt",
# "data/....csv", "output/..."), so the repo root must stay the working
# directory (WORKDIR) at run time.
COPY . /repo

# output/ receives the .rda files produced by fit_TKTD_bayes(); mount it as
# a volume at `docker run` time to persist results outside the container
# (see DEPLOY_SCALEWAY.md), otherwise they don't survive `docker run --rm`.
RUN mkdir -p /repo/output

# No fixed ENTRYPOINT: the image works both as an interactive shell
# (`docker run -it ... bash`) and for launching a specific script, e.g.
#   docker run --rm -v "$PWD/output:/repo/output" <image> \
#     Rscript run/run_TKTD_bayes.R --data data/data_artificial_additive.csv --bridge n_exp
CMD ["/bin/bash"]
