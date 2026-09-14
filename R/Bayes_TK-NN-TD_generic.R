## =============================================================================
## R/Bayes_TK-NN-TD_generic.R
## -----------------------------------------------------------------------------
## Generic version of the Bayesian TK-(NN)-TD-IT fitting engine shared by
## every R/Bayes_TKTD_*.R and R/Bayes_TK(linear|NN)TD*.R script in this repo.
##
## Those 11 scripts share ~95% of the same code (data preprocessing,
## building 'data_IT', jags.model()/update()/coda.samples()/WAIC, saving)
## and only differ by:
##   - which JAGS file is used (src/JAGS_*.txt)
##   - which parameters coda.samples() monitors
##   - the shape of the alpha prior (scalar vs. "split" vector)
##   - the output file name
##
## This script factors that logic into a single function, fit_TKTD_bayes(),
## which can be driven:
##   (a) programmatically, by sourcing this file and calling
##       fit_TKTD_bayes(...) with an already-loaded data.frame;
##   (b) from the command line via Rscript, which lets you test the model
##       against any CSV in data/ without editing a script:
##
##   Rscript "R/Bayes_TK-NN-TD_generic.R" \
##       --data data/MaXim__raw_datasets__set1_CLEAN.csv \
##       --mixture spiroxamine,prothioconazole,tebuconazole,trifloxystrobin,bixafen,fluopyram \
##       --bridge generic --hidden-layers 6 \
##       --id set1
##
## 'bridge' selects the architecture: "generic" (recommended -- see below)
## or one of the 11 values below, kept for cross-checking against the
## original per-architecture JAGS files, and listed in BRIDGE_REGISTRY:
##   NN, NN_split, linear, linear_split,
##   n, n_exp, nn_n_exp, nn_ReLU_n_exp,
##   nn_ReLU_nn_ReLU_n_exp, nn_ReLU_nn_ReLU_nn_ReLU_n_exp, n2n_ReLU_2n_exp
##
## bridge = "generic" uses the parametric JAGS model
## src/JAGS_TKTD_IT_generic.txt: a multilayer perceptron with free depth AND
## width (W_mask/b_mask connectivity masks built from the --hidden-layers
## vector), which also unifies the "split" mode (alpha estimated by the
## SAME network as the damage bridge, as in Bayes_TK(NN|linear)TD_split.R)
## instead of treating it as a separate case. Driven by
## --hidden-layers/--neg-slope/--out-exp/--alpha-split/--alpha-out-exp/
## --alpha-activation/--prec-w: every one of the 11 legacy architectures
## (and the paper's models, see DEPLOY_SCALEWAY.md) is reproducible this
## way, and so are architectures that match none of them, without writing a
## new JAGS file.
##
## IMPORTANT (precondition inherited from the original scripts): the rows of
## 'd' must be grouped into contiguous blocks by 'replicate' (each replicate
## on consecutive rows), and the first row of each replicate must have
## time == 0. This holds for every CSV in data/ (checked), but is NOT
## re-validated for a new file beyond the warning emitted below.
## =============================================================================

suppressMessages({
    library("coda")
    library("rjags")
})

## -----------------------------------------------------------------------------
## 1) Registry of "legacy" architectures (special cases of the generic engine)
## -----------------------------------------------------------------------------
## family = "A" -> output name "output/mcmc_<ID_SET>_JAGS_TK<bridge>TD_IT.rda"
##                 (convention used by the Bayes_TK(NN|linear)TD*.R scripts)
## family = "B" -> output name "output/mcmc_<ID_SET>_JAGS_TKTD_IT_<bridge>.rda"
##                 (convention used by the Bayes_TKTD_*.R scripts)
BRIDGE_REGISTRY <- list(
    "NN" = list(
        jags_file   = "src/JAGS_TKNNTD_IT.txt",
        monitor     = c("W1", "b1", "W2", "b2", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "A"
    ),
    "NN_split" = list(
        jags_file   = "src/JAGS_TKNNTD_split_IT.txt",
        monitor     = c("W1", "b1", "W2", "b2", "kd", "hb", "alpha_split", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = TRUE,
        family      = "A"
    ),
    "linear" = list(
        jags_file   = "src/JAGS_TKlinearTD_IT.txt",
        monitor     = c("a", "b1", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "A"
    ),
    "linear_split" = list(
        jags_file   = "src/JAGS_TKlinearTD_split_IT.txt",
        monitor     = c("a", "b1", "kd", "hb", "alpha_split", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = TRUE,
        family      = "A"
    ),
    "n" = list(
        jags_file   = "src/JAGS_TKTD_IT_n.txt",
        monitor     = c("a", "b1", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "B"
    ),
    "n_exp" = list(
        jags_file   = "src/JAGS_TKTD_IT_n_exp.txt",
        monitor     = c("a", "b1", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "B"
    ),
    "nn_n_exp" = list(
        jags_file   = "src/JAGS_TKTD_IT_nn_n_exp.txt",
        monitor     = c("W1", "b1", "W2", "b2", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "B"
    ),
    "nn_ReLU_n_exp" = list(
        jags_file   = "src/JAGS_TKTD_IT_nn_ReLU_n_exp.txt",
        monitor     = c("W1", "b1", "W2", "b2", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "B"
    ),
    ## NB: the original Bayes_TKTD_nn_ReLU_nn_ReLU_n_exp.R script does not
    ## monitor W3/b3 (2nd hidden layer) even though the model has them; fixed
    ## here so the trace actually covers every weight in the network.
    "nn_ReLU_nn_ReLU_n_exp" = list(
        jags_file   = "src/JAGS_TKTD_IT_nn_ReLU_nn_ReLU_n_exp.txt",
        monitor     = c("W1", "b1", "W2", "b2", "W3", "b3", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "B"
    ),
    ## same fix: W3/b3/W4/b4 added (missing from the original script)
    "nn_ReLU_nn_ReLU_nn_ReLU_n_exp" = list(
        jags_file   = "src/JAGS_TKTD_IT_nn_ReLU_nn_ReLU_nn_ReLU_n_exp.txt",
        monitor     = c("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "B"
    ),
    ## NB: src/JAGS_TKTD_IT_n2n_ReLU_2n_exp.txt has an R/JAGS precedence bug
    ## inherited from the original repo: `for(i in 1:2*m_layer)` is parsed as
    ## `for(i in (1:2)*m_layer)`, so the hidden layer does NOT actually have
    ## 2*m_layer neurons (it behaves like nn_ReLU_n_exp, with a couple of
    ## extra priors never connected to the likelihood). Not fixed here, to
    ## stay faithful to the original script; for a genuinely double-width
    ## network, use bridge = "generic" with --m-layer <2*N_EXPOSURE>.
    "n2n_ReLU_2n_exp" = list(
        jags_file   = "src/JAGS_TKTD_IT_n2n_ReLU_2n_exp.txt",
        monitor     = c("W1", "b1", "W2", "b2", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim"),
        alpha_split = FALSE,
        family      = "B"
    )
)

## -----------------------------------------------------------------------------
## 2) Data preprocessing (identical to the 11 original scripts)
## -----------------------------------------------------------------------------
.prepare_TKTD_data <- function(d, MIXTURE, N_EXPOSURE, m_layer,
                                kd_meanlog10, kd_sdlog10,
                                hb_meanlog10, hb_sdlog10,
                                alpha_meanlog10, alpha_sdlog10,
                                beta_minlog10, beta_maxlog10,
                                hb_value, hb_valueFIXED,
                                extra = list()) {

    ## -- safety check: each 'replicate' must form one contiguous block in 'd'
    rle_rep <- rle(as.character(d$replicate))
    if (length(rle_rep$values) != length(unique(d$replicate))) {
        warning(
            "Rows of 'd' are not grouped into contiguous blocks by ",
            "'replicate'. Nprec/i_prec are computed from row order and ",
            "assume 'd' is pre-sorted by replicate then by time: sort 'd' ",
            "(order(d$replicate, d$time)) before fitting."
        )
    }

    uniq_rep <- unique(d$replicate)
    ls_Nprec <- lapply(uniq_rep, function(r) {
        dat <- d[d$replicate == r, ]
        a <- dat$Nsurv
        c(a[1], a[-length(a)])
    })
    Nprec <- do.call("c", ls_Nprec)

    d$i_row <- seq_len(nrow(d))
    d$lag_i_row <- c(d$i_row[1], d$i_row[-length(d$i_row)])
    d$i_prec <- ifelse(d$time == 0, d$i_row, d$lag_i_row)
    iprec <- d$i_prec

    ls_replicate_ID <- lapply(seq_along(uniq_rep), function(i) {
        dat <- d[d$replicate == uniq_rep[i], ]
        rep(i, length(dat$replicate))
    })
    replicate_ID <- do.call("c", ls_replicate_ID)

    ls_time_ID <- lapply(uniq_rep, function(r) {
        dat <- d[d$replicate == r, ]
        seq_len(length(dat$time))
    })
    time_ID <- do.call("c", ls_time_ID)

    data_IT <- list(
        n_data = nrow(d),
        time = d$time,
        time_ID = time_ID,
        replicate_ID = replicate_ID,
        i_prec = iprec,
        Nsurv = d$Nsurv,
        Nprec = Nprec,
        n_X = N_EXPOSURE,
        X = matrix(MIXTURE, ncol = N_EXPOSURE, byrow = FALSE),
        kd_meanlog10 = rep(kd_meanlog10, length.out = N_EXPOSURE),
        kd_sdlog10 = rep(kd_sdlog10, length.out = N_EXPOSURE),
        hb_meanlog10 = hb_meanlog10,
        hb_sdlog10 = hb_sdlog10,
        alpha_meanlog10 = alpha_meanlog10,
        alpha_sdlog10 = alpha_sdlog10,
        beta_minlog10 = beta_minlog10,
        beta_maxlog10 = beta_maxlog10,
        hb_value = hb_value,
        hb_valueFIXED = hb_valueFIXED,
        # Neural Network
        m_layer = m_layer
    )
    modifyList(data_IT, extra)
}

## -----------------------------------------------------------------------------
## 2b) Building the masks for the generic bridge (bridge == "generic")
## -----------------------------------------------------------------------------
## hidden_layers: vector of hidden-layer widths, e.g. integer(0) (linear
##   n_X -> 1 bridge, equivalent to "n"/"n_exp"), c(4) (equivalent to
##   "nn_ReLU_n_exp" with m_layer = 4), c(4, 4), c(4, 4, 4), etc. -- no
##   depth limit, unlike the n_layers <= 3 cap in an earlier version of
##   this file.
.build_generic_bridge_data <- function(n_X, hidden_layers = integer(0), neg_slope = 0, prec_w = 4) {
    layer_sizes <- c(n_X, hidden_layers, 1)
    n_layer <- length(layer_sizes) - 1
    M <- max(layer_sizes)

    neg_slope <- rep(neg_slope, length.out = n_layer)

    ## W_mask[l,k,j] = 1 if layer l connects input j to output k;
    ## b_mask[l,k] = 1 if the bias of neuron k in layer l exists.
    ## Inputs/outputs beyond each layer's real size stay at 0 ("phantom"
    ## neurons, never connected to the likelihood).
    W_mask <- array(0, dim = c(n_layer, M, M))
    b_mask <- array(0, dim = c(n_layer, M))
    for (l in seq_len(n_layer)) {
        n_in  <- layer_sizes[l]
        n_out <- layer_sizes[l + 1]
        W_mask[l, seq_len(n_out), seq_len(n_in)] <- 1
        b_mask[l, seq_len(n_out)] <- 1
    }

    ## copy/mask the n_X real exposure columns onto the M columns of the
    ## network (columns n_X+1..M are zeroed out by X/in_mask, so the
    ## kd_idx/alpha_idx value for those columns has no effect)
    pad <- function(x) if (M > n_X) c(x, rep(1L, M - n_X)) else x
    kd_idx    <- pad(seq_len(n_X))
    alpha_idx <- pad(seq_len(n_X))
    in_mask   <- c(rep(1L, n_X), rep(0L, M - n_X))

    list(
        M = M, n_layer = n_layer,
        W_mask = W_mask, b_mask = b_mask,
        neg_slope = neg_slope,
        kd_idx = kd_idx, in_mask = in_mask, alpha_idx = alpha_idx,
        prec_w = prec_w
    )
}

## -----------------------------------------------------------------------------
## 3) Main function: generic equivalent of the 11 Bayes_*.R scripts
## -----------------------------------------------------------------------------
fit_TKTD_bayes <- function(
    d,
    MIXTURE,
    N_EXPOSURE,
    ID_SET,
    bridge,
    m_layer       = N_EXPOSURE, # only used for the 11 "legacy" bridges
    hidden_layers = integer(0), # only used if bridge == "generic"; e.g. c(4,4)
    neg_slope     = 0,          # same; 0 = ReLU, 1 = identity, 0.01 = leaky ReLU
    out_exp       = 1,          # same; 1 -> D = exp(bridge output), 0 -> D = raw output
    alpha_out_exp = 1,          # same, if alpha_split == TRUE; 1 -> alpha = exp(output)
    alpha_activation = 1,       # same; 0 reproduces Bayes_TKNNTD_split.R exactly (ReLU on the D branch, no activation at all on the alpha branch, despite sharing weights)
    prec_w        = 4,          # same; precision of the weight priors (default: dnorm(0,4), as in the 11 legacy scripts)
    jags_file   = NULL, # overrides the registry's JAGS file if provided
    monitor     = NULL, # overrides the monitored-parameter list if provided
    alpha_split = NULL, # bridge == "generic": TRUE = "split" alpha (same weights as the D bridge); otherwise overrides the registry
    n_update    = 5000,
    n_iter      = 2500,
    n_iter_waic = 1500,
    n_chains    = 3,
    kd_meanlog10 = -1.5, kd_sdlog10 = 0.5,
    hb_meanlog10 = -1.5, hb_sdlog10 = 0.5,
    alpha_meanlog10 = -1.5, alpha_sdlog10 = 0.5,
    beta_minlog10 = -2, beta_maxlog10 = 2,
    hb_value = 1, hb_valueFIXED = 0,
    output_dir  = "output",
    save_output = TRUE,
    verbose     = TRUE,
    ## MLflow experiment tracking (see .mlflow_log_run() below): auto-enabled
    ## when MLFLOW_TRACKING_URI is set in the environment (e.g. by `docker
    ## run -e MLFLOW_TRACKING_URI=...`), unless explicitly overridden here.
    mlflow_tracking    = NULL,
    mlflow_experiment  = "TKTD-NeuralODEs"
) {
    is_generic <- identical(bridge, "generic")

    if (is_generic) {
        if (is.null(jags_file))   jags_file   <- "src/JAGS_TKTD_IT_generic.txt"
        if (is.null(alpha_split)) alpha_split <- FALSE
        bridge_data <- .build_generic_bridge_data(
            n_X = N_EXPOSURE, hidden_layers = hidden_layers,
            neg_slope = neg_slope, prec_w = prec_w
        )
        if (is.null(monitor)) {
            ## W/b group ALL layers ([n_layer,M,M]/[n_layer,M] arrays,
            ## including weights masked to 0): more verbose than monitoring
            ## layer by layer, but correct whatever depth was chosen.
            monitor <- c("W", "b", "kd", "hb", "alpha", "beta", "Nsurv_ppc", "Nsurv_sim")
        }
        out_tag <- sprintf(
            "JAGS_TKTD_IT_generic_L%d_M%d_out%d_alphasplit%d_aact%d",
            bridge_data$n_layer, bridge_data$M, out_exp, as.integer(isTRUE(alpha_split)), as.integer(isTRUE(alpha_activation))
        )
    } else {
        if (!bridge %in% names(BRIDGE_REGISTRY)) {
            stop(sprintf(
                "Unknown bridge = '%s'. Possible values: %s, or 'generic'.",
                bridge, paste(names(BRIDGE_REGISTRY), collapse = ", ")
            ))
        }
        spec <- BRIDGE_REGISTRY[[bridge]]
        if (is.null(jags_file))   jags_file   <- spec$jags_file
        if (is.null(monitor))     monitor     <- spec$monitor
        if (is.null(alpha_split)) alpha_split <- spec$alpha_split
        out_tag <- if (identical(spec$family, "A")) {
            sprintf("JAGS_TK%sTD_IT", bridge)
        } else {
            sprintf("JAGS_TKTD_IT_%s", bridge)
        }
    }

    ## alpha prior: scalar, or a vector of length N_EXPOSURE for the
    ## "split" architectures (alpha estimated from one alpha per substance).
    ## The "generic" bridge ALWAYS declares alpha_log10[1:n_X] in the JAGS
    ## file (the alpha_split branch is evaluated unconditionally there), so
    ## the prior must always be supplied there as a vector of size n_X, even
    ## when alpha_split == FALSE (only alpha_log10[1] is then used).
    if (isTRUE(alpha_split) || is_generic) {
        alpha_meanlog10_ <- rep(alpha_meanlog10, length.out = N_EXPOSURE)
        alpha_sdlog10_   <- rep(alpha_sdlog10,   length.out = N_EXPOSURE)
    } else {
        alpha_meanlog10_ <- alpha_meanlog10[1]
        alpha_sdlog10_   <- alpha_sdlog10[1]
    }

    extra <- list()
    if (is_generic) {
        X_padded <- matrix(0, nrow = nrow(d), ncol = bridge_data$M)
        X_padded[, seq_len(N_EXPOSURE)] <- matrix(MIXTURE, ncol = N_EXPOSURE, byrow = FALSE)
        extra <- list(
            X = X_padded,
            M = bridge_data$M, n_layer = bridge_data$n_layer,
            W_mask = bridge_data$W_mask, b_mask = bridge_data$b_mask,
            neg_slope = bridge_data$neg_slope,
            kd_idx = bridge_data$kd_idx, in_mask = bridge_data$in_mask, alpha_idx = bridge_data$alpha_idx,
            prec_w = bridge_data$prec_w,
            out_exp = out_exp,
            alpha_split = as.integer(isTRUE(alpha_split)),
            alpha_out_exp = as.integer(isTRUE(alpha_out_exp)),
            alpha_activation = as.integer(isTRUE(alpha_activation))
        )
    }

    data_IT <- .prepare_TKTD_data(
        d = d, MIXTURE = MIXTURE, N_EXPOSURE = N_EXPOSURE, m_layer = m_layer,
        kd_meanlog10 = kd_meanlog10, kd_sdlog10 = kd_sdlog10,
        hb_meanlog10 = hb_meanlog10, hb_sdlog10 = hb_sdlog10,
        alpha_meanlog10 = alpha_meanlog10_, alpha_sdlog10 = alpha_sdlog10_,
        beta_minlog10 = beta_minlog10, beta_maxlog10 = beta_maxlog10,
        hb_value = hb_value, hb_valueFIXED = hb_valueFIXED,
        extra = extra
    )

    ### Model building
    if (!file.exists(jags_file)) {
        stop(sprintf(
            "JAGS file not found: '%s' (working directory: %s). Run this script from the repo root (the one containing the R/, src/ and data/ folders).",
            jags_file, getwd()
        ))
    }
    if (verbose) print(paste("START MODEL", ID_SET, bridge))
    model <- jags.model(
        file = jags_file,
        data = data_IT,
        n.chains = n_chains
    )

    if (verbose) print(paste("START UPDATE", ID_SET, bridge))
    update(model, n_update)

    if (verbose) print(paste("START TRACE", ID_SET, bridge))
    mcmc_trace <- coda.samples(
        model,
        monitor,
        n.iter = n_iter
    )

    if (verbose) print(paste("START WAIC", ID_SET, bridge))
    load.module("dic")
    mcmc_mean <- jags.samples(
        model, c("deviance", "WAIC"),
        type = "mean", n.iter = n_iter_waic
    )

    result <- list(model = model, mcmc_trace = mcmc_trace, mcmc_mean = mcmc_mean,
                    d = d, data_IT = data_IT, jags_file = jags_file, monitor = monitor)

    if (save_output) {
        if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
        path <- file.path(output_dir, sprintf("mcmc_%s_%s.rda", ID_SET, out_tag))
        save(file = path, model, mcmc_mean, mcmc_trace, d)
        if (verbose) print(paste("SAVED", path))
        result$output_path <- path
    }

    mlflow_enabled <- if (is.null(mlflow_tracking)) {
        nzchar(Sys.getenv("MLFLOW_TRACKING_URI"))
    } else {
        isTRUE(mlflow_tracking)
    }
    if (mlflow_enabled) {
        params <- list(
            bridge = bridge, id_set = ID_SET, n_exposure = N_EXPOSURE,
            n_update = n_update, n_iter = n_iter, n_iter_waic = n_iter_waic, n_chains = n_chains
        )
        if (is_generic) {
            params <- c(params, list(
                hidden_layers = paste(hidden_layers, collapse = ","),
                n_layer = bridge_data$n_layer, m = bridge_data$M,
                neg_slope = paste(neg_slope, collapse = ","),
                out_exp = out_exp, alpha_split = as.integer(isTRUE(alpha_split)),
                alpha_out_exp = as.integer(isTRUE(alpha_out_exp)),
                alpha_activation = as.integer(isTRUE(alpha_activation)),
                prec_w = prec_w
            ))
        } else {
            params <- c(params, list(m_layer = m_layer))
        }
        metrics <- list(
            mean_deviance = tryCatch(mean(mcmc_mean$deviance), error = function(e) NA_real_),
            mean_WAIC     = tryCatch(mean(mcmc_mean$WAIC), error = function(e) NA_real_)
        )
        .mlflow_log_run(
            experiment = mlflow_experiment,
            run_name = sprintf("%s_%s", ID_SET, out_tag),
            params = params,
            metrics = metrics,
            artifact_path = result$output_path,
            verbose = verbose
        )
    }

    invisible(result)
}

## -----------------------------------------------------------------------------
## 3b) Optional MLflow logging (used when MLFLOW_TRACKING_URI is set, or
## mlflow_tracking = TRUE is passed to fit_TKTD_bayes()/run_TKTD_simulation())
## -----------------------------------------------------------------------------
## Best-effort only: this never aborts a fit. If the 'mlflow' R package is
## missing, or the tracking server is unreachable, a warning is emitted and
## the (already-completed) fit result is still returned untouched.
.mlflow_log_run <- function(experiment, run_name, params, metrics, artifact_path, verbose = TRUE) {
    if (!requireNamespace("mlflow", quietly = TRUE)) {
        warning(
            "MLFLOW_TRACKING_URI is set but the 'mlflow' R package is not installed; ",
            "skipping experiment tracking for this run. Install it with ",
            "install.packages('mlflow') (needs Python mlflow reachable via ",
            "MLFLOW_TRACKING_URI; see DEPLOY_SCALEWAY.md)."
        )
        return(invisible(NULL))
    }
    tryCatch({
        mlflow::mlflow_set_experiment(experiment)
        mlflow::mlflow_start_run(run_name = run_name)
        on.exit(mlflow::mlflow_end_run(), add = TRUE)
        for (nm in names(params)) {
            mlflow::mlflow_log_param(nm, as.character(params[[nm]]))
        }
        for (nm in names(metrics)) {
            if (!is.na(metrics[[nm]])) mlflow::mlflow_log_metric(nm, as.numeric(metrics[[nm]]))
        }
        if (!is.null(artifact_path) && file.exists(artifact_path)) {
            mlflow::mlflow_log_artifact(artifact_path)
        }
        if (verbose) print(sprintf("MLflow: logged run '%s' to experiment '%s'", run_name, experiment))
    }, error = function(e) {
        warning(sprintf("MLflow logging failed (fit result is unaffected): %s", conditionMessage(e)))
    })
}

## -----------------------------------------------------------------------------
## 4) Rscript (CLI) entry point: Rscript "R/Bayes_TK-NN-TD_generic.R" --data ... --bridge ... --mixture ...
## -----------------------------------------------------------------------------
.parse_cli_args <- function(args) {
    out <- list()
    i <- 1
    while (i <= length(args)) {
        a <- args[[i]]
        if (startsWith(a, "--")) {
            key <- substring(a, 3)
            if (i < length(args) && !startsWith(args[[i + 1]], "--")) {
                out[[key]] <- args[[i + 1]]
                i <- i + 2
            } else {
                out[[key]] <- "TRUE"
                i <- i + 1
            }
        } else {
            i <- i + 1
        }
    }
    out
}

## TRUE only when THIS file is the literal script passed to `Rscript ...`
## (not merely sourced by another script), so that source()-ing this file
## from e.g. run/run_TKTD_bayes.R never re-triggers its own CLI block.
.is_rscript_main <- function(self_basename) {
    a <- commandArgs(trailingOnly = FALSE)
    if (interactive()) return(FALSE)
    file_arg <- a[grepl("^--file=", a)]
    if (length(file_arg) == 0) return(FALSE)
    identical(basename(sub("^--file=", "", file_arg[[1]])), self_basename)
}

if (.is_rscript_main("Bayes_TK-NN-TD_generic.R")) {

    .cli_args <- .parse_cli_args(commandArgs(trailingOnly = TRUE))

    .get_arg <- function(name, default = NULL, type = "character") {
        v <- .cli_args[[name]]
        if (is.null(v)) {
            if (is.null(default)) {
                stop(sprintf("Missing required argument --%s.", name))
            }
            v <- default
        }
        switch(type,
            integer = as.integer(v),
            numeric = as.numeric(v),
            v
        )
    }

    data_path <- .get_arg("data")
    bridge    <- .get_arg("bridge")
    mixture_cols <- trimws(strsplit(.get_arg("mixture"), ",")[[1]])
    sep       <- .get_arg("sep", ",")
    id_set    <- .get_arg("id", tools::file_path_sans_ext(basename(data_path)))

    n_update    <- .get_arg("n-update", 5000, "integer")
    n_iter      <- .get_arg("n-iter", 2500, "integer")
    n_iter_waic <- .get_arg("n-iter-waic", 1500, "integer")
    n_chains    <- .get_arg("n-chains", 3, "integer")
    output_dir  <- .get_arg("output-dir", "output")

    kd_meanlog10    <- .get_arg("kd-meanlog10", -1.5, "numeric")
    kd_sdlog10      <- .get_arg("kd-sdlog10", 0.5, "numeric")
    hb_meanlog10    <- .get_arg("hb-meanlog10", -1.5, "numeric")
    hb_sdlog10      <- .get_arg("hb-sdlog10", 0.5, "numeric")
    alpha_meanlog10 <- .get_arg("alpha-meanlog10", -1.5, "numeric")
    alpha_sdlog10   <- .get_arg("alpha-sdlog10", 0.5, "numeric")
    beta_minlog10   <- .get_arg("beta-minlog10", -2, "numeric")
    beta_maxlog10   <- .get_arg("beta-maxlog10", 2, "numeric")
    hb_value        <- .get_arg("hb-value", 1, "integer")
    hb_valueFIXED   <- .get_arg("hb-valuefixed", 0, "numeric")

    d <- read.delim(data_path, header = TRUE, sep = sep)

    mixture_cols <- make.names(mixture_cols)
    missing_cols <- setdiff(mixture_cols, colnames(d))
    if (length(missing_cols) > 0) {
        stop(sprintf(
            "Mixture columns not found in %s: %s\nAvailable columns: %s",
            data_path, paste(missing_cols, collapse = ", "), paste(colnames(d), collapse = ", ")
        ))
    }

    N_EXPOSURE <- length(mixture_cols)
    MIXTURE <- do.call("c", lapply(mixture_cols, function(cn) d[[cn]]))

    m_layer <- .get_arg("m-layer", N_EXPOSURE, "integer")

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
    ## --alpha-split only makes sense for bridge == "generic": for the
    ## legacy bridges, NULL lets BRIDGE_REGISTRY decide (e.g.
    ## "NN_split"/"linear_split" force TRUE regardless of this argument).
    alpha_split_arg <- .cli_args[["alpha-split"]]
    alpha_split <- if (identical(bridge, "generic")) {
        if (is.null(alpha_split_arg)) FALSE else as.logical(as.integer(alpha_split_arg))
    } else {
        NULL
    }
    mlflow_experiment <- .get_arg("mlflow-experiment", "TKTD-NeuralODEs")

    fit_TKTD_bayes(
        d = d, MIXTURE = MIXTURE, N_EXPOSURE = N_EXPOSURE, ID_SET = id_set,
        bridge = bridge, m_layer = m_layer,
        hidden_layers = hidden_layers, neg_slope = neg_slope,
        out_exp = out_exp, alpha_split = alpha_split, alpha_out_exp = alpha_out_exp,
        alpha_activation = alpha_activation, prec_w = prec_w,
        n_update = n_update, n_iter = n_iter, n_iter_waic = n_iter_waic, n_chains = n_chains,
        kd_meanlog10 = kd_meanlog10, kd_sdlog10 = kd_sdlog10,
        hb_meanlog10 = hb_meanlog10, hb_sdlog10 = hb_sdlog10,
        alpha_meanlog10 = alpha_meanlog10, alpha_sdlog10 = alpha_sdlog10,
        beta_minlog10 = beta_minlog10, beta_maxlog10 = beta_maxlog10,
        mlflow_experiment = mlflow_experiment,
        hb_value = hb_value, hb_valueFIXED = hb_valueFIXED,
        output_dir = output_dir
    )
}
