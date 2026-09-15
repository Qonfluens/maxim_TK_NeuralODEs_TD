## =============================================================================
## python/summarize_jags_rda.R
## -----------------------------------------------------------------------------
## Condenses the JAGS .rda outputs into one tidy CSV so the PyTorch results can
## be compared against them without loading R again.
##
## For every output/mcmc_*.rda it extracts the posterior mean, sd and 95%
## credible interval of the interpretable parameters (kd, alpha, beta, hb),
## plus the mean deviance and mean WAIC that fit_TKTD_bayes() recorded.
##
## The network weights are deliberately NOT summarised: a ReLU bridge is
## invariant to permuting hidden units and to rescaling a layer against the
## next one, so individual weights are not comparable between two fits, let
## alone between two inference methods. Compare kd/alpha/beta/hb, the fitted
## survival curves, and the ranking of architectures instead.
##
## Usage (from the repo root):
##   Rscript python/summarize_jags_rda.R [rda_dir] [out_csv]
##   # defaults: output/  ->  output/jags_summary.csv
## =============================================================================

suppressMessages({
    library("coda")
})

args <- commandArgs(trailingOnly = TRUE)
rda_dir <- if (length(args) > 0) args[1] else "output"
out_csv <- if (length(args) > 1) args[2] else file.path(rda_dir, "jags_summary.csv")

files <- list.files(rda_dir, pattern = "^mcmc_.*\\.rda$", full.names = TRUE)
if (length(files) == 0) {
    stop(sprintf("No mcmc_*.rda files found in '%s'.", rda_dir))
}
cat(sprintf("Summarising %d .rda file(s) from %s\n", length(files), rda_dir))

rows <- list()

for (f in files) {
    env <- new.env()
    ok <- tryCatch({ load(f, envir = env); TRUE },
                   error = function(e) { cat("  SKIP", basename(f), ":", conditionMessage(e), "\n"); FALSE })
    if (!ok) next

    ## mcmc_<id_set>_<tag>.rda -> id_set and tag
    stem <- sub("\\.rda$", "", basename(f))
    stem <- sub("^mcmc_", "", stem)
    tag <- regmatches(stem, regexpr("JAGS_TKTD_IT_generic_L[0-9]+_M[0-9]+_out[0-9]+_alphasplit[0-9]+(_aact[0-9]+)?$", stem))
    id_set <- if (length(tag) == 1) sub(sprintf("_%s$", tag), "", stem) else stem
    if (length(tag) != 1) tag <- NA_character_

    tr <- tryCatch(as.matrix(env$mcmc_trace), error = function(e) NULL)
    if (is.null(tr)) { cat("  SKIP", basename(f), ": no usable mcmc_trace\n"); next }

    keep <- grep("^(kd\\[[0-9]+\\]|alpha|beta|hb)$", colnames(tr))
    if (length(keep) == 0) { cat("  SKIP", basename(f), ": no kd/alpha/beta/hb monitored\n"); next }

    for (j in keep) {
        v <- tr[, j]
        q <- unname(quantile(v, c(0.025, 0.975)))
        rows[[length(rows) + 1]] <- data.frame(
            file = basename(f), id_set = id_set, tag = tag,
            parameter = colnames(tr)[j],
            mean = mean(v), sd = sd(v), q2.5 = q[1], q97.5 = q[2],
            stringsAsFactors = FALSE
        )
    }

    add_metric <- function(name, value) {
        rows[[length(rows) + 1]] <<- data.frame(
            file = basename(f), id_set = id_set, tag = tag, parameter = name,
            mean = value, sd = NA_real_, q2.5 = NA_real_, q97.5 = NA_real_,
            stringsAsFactors = FALSE
        )
    }

    ## NOTE: the dic module's 'deviance' and 'WAIC' monitors are PER OBSERVED
    ## NODE, so mean() over them (what fit_TKTD_bayes logs to MLflow as
    ## mean_deviance / mean_WAIC) is an average per observation, not a total.
    ## Both are recorded here; the total is what compares to a deviance
    ## computed as -2 * log-likelihood over the whole data set.
    n_data <- if (!is.null(env$d)) nrow(env$d) else NA_integer_
    add_metric("n_data", n_data)
    if (!is.null(env$mcmc_mean$deviance)) {
        add_metric("mean_deviance", mean(env$mcmc_mean$deviance))
        add_metric("total_deviance", sum(env$mcmc_mean$deviance))
    }
    if (!is.null(env$mcmc_mean$WAIC)) {
        add_metric("mean_WAIC", mean(env$mcmc_mean$WAIC))
        add_metric("total_WAIC", sum(env$mcmc_mean$WAIC))
    }

    ## -- posterior-mean fitted survival, the comparison that is actually
    ## identifiable (unlike alpha and beta individually, see the header note)
    ppc_cols <- grep("^Nsurv_ppc\\[[0-9]+\\]$", colnames(tr))
    if (length(ppc_cols) > 0) {
        idx <- as.integer(sub("^Nsurv_ppc\\[([0-9]+)\\]$", "\\1", colnames(tr)[ppc_cols]))
        pred <- data.frame(row = idx, Nsurv_ppc_mean = colMeans(tr[, ppc_cols, drop = FALSE]))
        pred <- pred[order(pred$row), ]
        pred_file <- file.path(dirname(out_csv), sprintf("jagspred_%s.csv", stem))
        write.csv(pred, pred_file, row.names = FALSE)
    }

    cat("  ok  ", basename(f), "\n")
}

if (length(rows) == 0) stop("Nothing could be summarised.")

res <- do.call(rbind, rows)
write.csv(res, out_csv, row.names = FALSE)
cat(sprintf("\nWrote %d rows to %s\n", nrow(res), out_csv))
cat("Now run: python3 python/compare_jags_torch.py", out_csv, "output_torch\n")
