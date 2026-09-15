## =============================================================================
## python/validate_against_jags.R
## -----------------------------------------------------------------------------
## Step 1 of the JAGS <-> PyTorch equivalence check.
##
## Runs the real JAGS pipeline (fit_TKTD_bayes, unchanged) for a few
## architectures, then dumps to CSV:
##   - the preprocessed data arrays JAGS was actually given (data_IT), so the
##     Python preprocessing can be checked against it row by row;
##   - one MCMC iteration's parameter values TOGETHER with the deviance JAGS
##     computed at exactly those values.
##
## Step 2 (python/validate_against_jags.py) loads those values into the PyTorch
## model and checks it returns the same deviance. Matching to ~1e-8 proves the
## port is exact, which a comparison of two independent fits never could.
##
## Usage (from the repo root):
##   Rscript python/validate_against_jags.R [output_dir]
## =============================================================================

suppressMessages({
    library("coda")
    library("rjags")
})

out_dir <- if (length(commandArgs(trailingOnly = TRUE)) > 0) {
    commandArgs(trailingOnly = TRUE)[1]
} else {
    "output/validation"
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("R/Bayes_TK-NN-TD_generic.R", chdir = FALSE)

## the 'deviance' node only exists once the dic module is loaded, and it must
## be loaded BEFORE coda.samples() is called
load.module("dic")

DATA_PATH <- "data/data_artificial_additive.csv"
MIXTURE_COLS <- c("A", "B")

d <- read.delim(DATA_PATH, header = TRUE, sep = ",")
N_EXPOSURE <- length(MIXTURE_COLS)
MIXTURE <- do.call("c", lapply(MIXTURE_COLS, function(cn) d[[cn]]))

## architectures covering: linear bridge / one hidden ReLU layer / two hidden
## layers / identity activation. All with alpha_split = 0, because JAGS
## monitors the scalar 'alpha' and that is only invertible to alpha_log10[1]
## in the non-split case.
CASES <- list(
    list(name = "n_exp",                 hidden = integer(0), neg_slope = 0, out_exp = 1),
    list(name = "n",                     hidden = integer(0), neg_slope = 0, out_exp = 0),
    list(name = "nn_ReLU_n_exp",         hidden = c(2),       neg_slope = 0, out_exp = 1),
    list(name = "nn_n_exp",              hidden = c(2),       neg_slope = 1, out_exp = 1),
    list(name = "nn_ReLU_nn_ReLU_n_exp", hidden = c(2, 2),    neg_slope = 0, out_exp = 1)
)

for (case in CASES) {
    cat("=== JAGS run:", case$name, "\n")
    res <- fit_TKTD_bayes(
        d = d, MIXTURE = MIXTURE, N_EXPOSURE = N_EXPOSURE, ID_SET = "validate",
        bridge = "generic",
        hidden_layers = case$hidden, neg_slope = case$neg_slope,
        out_exp = case$out_exp, alpha_split = 0, alpha_out_exp = 1,
        alpha_activation = 1,
        monitor = c("W", "b", "kd", "hb", "alpha", "beta", "deviance"),
        n_update = 200, n_iter = 20, n_iter_waic = 10, n_chains = 1,
        save_output = FALSE, verbose = FALSE
    )

    ## -- one iteration's parameters + the deviance JAGS computed there
    tr <- as.matrix(res$mcmc_trace[[1]])
    ## take the last iteration (furthest from the initial values)
    it <- tr[nrow(tr), ]
    write.csv(
        data.frame(name = names(it), value = as.numeric(it)),
        file.path(out_dir, sprintf("params_%s.csv", case$name)),
        row.names = FALSE
    )

    ## -- the preprocessed arrays JAGS was given
    dat <- res$data_IT
    write.csv(
        data.frame(
            time = dat$time, Nsurv = dat$Nsurv, Nprec = dat$Nprec,
            i_prec = dat$i_prec, time_ID = dat$time_ID,
            replicate_ID = dat$replicate_ID
        ),
        file.path(out_dir, sprintf("data_%s.csv", case$name)),
        row.names = FALSE
    )
    write.csv(as.data.frame(dat$X),
              file.path(out_dir, sprintf("X_%s.csv", case$name)), row.names = FALSE)

    ## -- the architecture flags, so Python rebuilds the identical bridge
    write.csv(
        data.frame(
            key = c("n_layer", "M", "n_X", "out_exp", "alpha_split",
                    "alpha_out_exp", "alpha_activation", "prec_w",
                    "neg_slope_1", "hidden_len"),
            value = c(dat$n_layer, dat$M, dat$n_X, dat$out_exp, dat$alpha_split,
                      dat$alpha_out_exp, dat$alpha_activation, dat$prec_w,
                      dat$neg_slope[1], length(case$hidden))
        ),
        file.path(out_dir, sprintf("cfg_%s.csv", case$name)), row.names = FALSE
    )
    cat("    deviance at dumped iteration:", it[["deviance"]], "\n")
}

cat("\nWrote validation dumps to", out_dir, "\n")
cat("Now run: python3 python/validate_against_jags.py", out_dir, "\n")
