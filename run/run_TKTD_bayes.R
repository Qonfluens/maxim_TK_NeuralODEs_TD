## =============================================================================
## run/run_TKTD_bayes.R
## -----------------------------------------------------------------------------
## Turnkey Rscript function: launches a Bayesian TK-(NN)-TD-IT fit from a
## SINGLE required argument (the path to a CSV in data/) by automatically
## inferring the exposure columns (the "MIXTURE" and "N_EXPOSURE" that the
## legacy R/run_all*.R scripts used to set by hand). It relies on the
## generic engine fit_TKTD_bayes() from R/Bayes_TK-NN-TD_generic.R (to avoid
## duplicating the fitting logic), but remains a standalone entry point,
## usable/callable independently of that file.
##
## Usage (from the repo root):
##
##   Rscript run/run_TKTD_bayes.R --data data/MaXim__raw_datasets__set1_CLEAN.csv
##   # (defaults to --bridge generic with no hidden layer and --out-exp 1,
##   # i.e. the paper's "n_exp" model -- see DEPLOY_SCALEWAY.md section 7.2)
##
##   Rscript run/run_TKTD_bayes.R \
##       --data data/data_artificial_synergism.csv \
##       --bridge generic --hidden-layers 2,2 \
##       --id synergism_test \
##       --n-update 1000 --n-iter 1000 --n-iter-waic 500
##
##   # split alpha (see DEPLOY_SCALEWAY.md for the full paper-reproduction table):
##   Rscript run/run_TKTD_bayes.R \
##       --data data/MaXim__raw_datasets__set1_CLEAN.csv \
##       --bridge generic --hidden-layers 6 --alpha-split 1
##
## Exposure columns: by default, any numeric CSV column that isn't in the
## NON_MIXTURE_COLUMNS list below is treated as an exposure
## substance/route (mixture). For an unusual CSV, pass
## --mixture col1,col2,... to force the list explicitly (as with
## R/Bayes_TK-NN-TD_generic.R).
## =============================================================================

suppressMessages({
    library("coda")
    library("rjags")
})

## loads fit_TKTD_bayes(), BRIDGE_REGISTRY, .parse_cli_args(), etc. without
## re-triggering its CLI block (that only activates when THAT file is the
## main Rscript file, see .is_rscript_main()).
.generic_script <- file.path(
    if (nzchar(Sys.getenv("TKTD_R_DIR"))) Sys.getenv("TKTD_R_DIR") else "R",
    "Bayes_TK-NN-TD_generic.R"
)
if (!file.exists(.generic_script)) {
    stop(sprintf(
        "Not found: '%s' (working directory: %s). Run this script from the repo root (the one containing the R/, src/ and data/ folders).",
        .generic_script, getwd()
    ))
}
source(.generic_script, chdir = FALSE)

## -----------------------------------------------------------------------------
## Columns that are never exposure substances/routes
## -----------------------------------------------------------------------------
NON_MIXTURE_COLUMNS <- c(
    "X", "time", "file", "replicate", "Nsurv", "Nprec",
    "unit_ai", "M_number", "Mixture_abcSorted", "formulation",
    "Psurv", "Interaction", "i_row", "lag_i_row", "i_prec"
)

infer_mixture_columns <- function(d, exclude = NON_MIXTURE_COLUMNS) {
    candidates <- setdiff(colnames(d), exclude)
    is_numeric <- vapply(d[candidates], is.numeric, logical(1))
    mixture_cols <- candidates[is_numeric]
    if (length(mixture_cols) == 0) {
        stop(
            "No exposure column could be auto-detected. ",
            "Available columns: ", paste(colnames(d), collapse = ", "),
            ". Use --mixture col1,col2,... to specify them explicitly."
        )
    }
    mixture_cols
}

## -----------------------------------------------------------------------------
## Main function: reads the CSV, infers MIXTURE/N_EXPOSURE, launches the fit
## -----------------------------------------------------------------------------
run_TKTD_simulation <- function(
    data_path,
    bridge       = "generic",
    mixture_cols = NULL,     # NULL = auto-detection
    id_set       = tools::file_path_sans_ext(basename(data_path)),
    sep          = ",",
    m_layer      = NULL,     # NULL = length(mixture_cols) (legacy bridges only)
    hidden_layers = integer(0), # bridge == "generic" only; e.g. c(4,4)
    neg_slope     = 0,
    out_exp       = 1,
    alpha_split   = NULL,
    alpha_out_exp = 1,
    alpha_activation = 1,
    prec_w        = 4,
    n_update     = 5000,
    n_iter       = 2500,
    n_iter_waic  = 1500,
    n_chains     = 3,
    output_dir   = "output",
    ...
) {
    if (!file.exists(data_path)) {
        stop(sprintf("Data file not found: %s", data_path))
    }
    d <- read.delim(data_path, header = TRUE, sep = sep)

    if (is.null(mixture_cols)) {
        mixture_cols <- infer_mixture_columns(d)
        message(sprintf(
            "Auto-detected exposure columns for %s: %s",
            basename(data_path), paste(mixture_cols, collapse = ", ")
        ))
    } else {
        mixture_cols <- make.names(trimws(mixture_cols))
        missing_cols <- setdiff(mixture_cols, colnames(d))
        if (length(missing_cols) > 0) {
            stop(sprintf(
                "Mixture columns not found in %s: %s\nAvailable columns: %s",
                data_path, paste(missing_cols, collapse = ", "), paste(colnames(d), collapse = ", ")
            ))
        }
    }

    N_EXPOSURE <- length(mixture_cols)
    MIXTURE <- do.call("c", lapply(mixture_cols, function(cn) d[[cn]]))
    if (is.null(m_layer)) m_layer <- N_EXPOSURE

    fit_TKTD_bayes(
        d = d, MIXTURE = MIXTURE, N_EXPOSURE = N_EXPOSURE, ID_SET = id_set,
        bridge = bridge, m_layer = m_layer,
        hidden_layers = hidden_layers, neg_slope = neg_slope,
        out_exp = out_exp, alpha_split = alpha_split, alpha_out_exp = alpha_out_exp,
        alpha_activation = alpha_activation, prec_w = prec_w,
        n_update = n_update, n_iter = n_iter, n_iter_waic = n_iter_waic, n_chains = n_chains,
        output_dir = output_dir,
        ...
    )
}

## -----------------------------------------------------------------------------
## Rscript entry point
## -----------------------------------------------------------------------------
if (.is_rscript_main("run_TKTD_bayes.R")) {

    .cli_args <- .parse_cli_args(commandArgs(trailingOnly = TRUE))

    .get_arg <- function(name, default = NULL, type = "character") {
        v <- .cli_args[[name]]
        if (is.null(v)) {
            if (is.null(default)) stop(sprintf("Missing required argument --%s.", name))
            v <- default
        }
        switch(type, integer = as.integer(v), numeric = as.numeric(v), v)
    }

    data_path <- .get_arg("data")
    bridge    <- .get_arg("bridge", "generic")
    mixture_arg <- .cli_args[["mixture"]]
    mixture_cols <- if (is.null(mixture_arg)) NULL else strsplit(mixture_arg, ",")[[1]]
    id_set    <- .get_arg("id", tools::file_path_sans_ext(basename(data_path)))
    sep       <- .get_arg("sep", ",")

    n_update    <- .get_arg("n-update", 5000, "integer")
    n_iter      <- .get_arg("n-iter", 2500, "integer")
    n_iter_waic <- .get_arg("n-iter-waic", 1500, "integer")
    n_chains    <- .get_arg("n-chains", 3, "integer")
    output_dir  <- .get_arg("output-dir", "output")

    m_layer_arg <- .cli_args[["m-layer"]]
    m_layer <- if (is.null(m_layer_arg)) NULL else as.integer(m_layer_arg)

    ## -- "generic" bridge options (ignored for the 11 legacy bridges)
    hidden_layers_arg <- .cli_args[["hidden-layers"]]
    hidden_layers <- if (is.null(hidden_layers_arg) || !nzchar(hidden_layers_arg)) {
        integer(0)
    } else {
        as.integer(strsplit(hidden_layers_arg, ",")[[1]])
    }
    neg_slope        <- .get_arg("neg-slope", 0, "numeric")
    out_exp          <- .get_arg("out-exp", 1, "integer")
    alpha_out_exp    <- .get_arg("alpha-out-exp", 1, "integer")
    alpha_activation <- .get_arg("alpha-activation", 1, "integer")
    prec_w           <- .get_arg("prec-w", 4, "numeric")
    alpha_split_arg <- .cli_args[["alpha-split"]]
    alpha_split <- if (identical(bridge, "generic")) {
        if (is.null(alpha_split_arg)) FALSE else as.logical(as.integer(alpha_split_arg))
    } else {
        NULL
    }
    mlflow_experiment <- .get_arg("mlflow-experiment", "TKTD-NeuralODEs")

    run_TKTD_simulation(
        data_path = data_path,
        bridge = bridge,
        mixture_cols = mixture_cols,
        id_set = id_set,
        sep = sep,
        m_layer = m_layer,
        hidden_layers = hidden_layers, neg_slope = neg_slope,
        out_exp = out_exp, alpha_split = alpha_split, alpha_out_exp = alpha_out_exp,
        alpha_activation = alpha_activation, prec_w = prec_w,
        n_update = n_update, n_iter = n_iter, n_iter_waic = n_iter_waic, n_chains = n_chains,
        output_dir = output_dir,
        mlflow_experiment = mlflow_experiment
    )
}
