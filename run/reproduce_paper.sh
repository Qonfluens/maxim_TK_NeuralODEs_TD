#!/usr/bin/env bash
# =============================================================================
# run/reproduce_paper.sh
# -----------------------------------------------------------------------------
# Reproduces every calibration run of the paper (Baudrot et al., PLOS
# Computational Biology 2025, Table 1 + the "split" alpha variants of
# Table 3) using ONLY the generic bridge (src/JAGS_TKTD_IT_generic.txt via
# run/run_TKTD_bayes.R --bridge generic), dispatched in parallel with GNU
# parallel (one job per CPU core by default) -- meant for a large multi-core
# Scaleway instance. See DEPLOY_SCALEWAY.md for the full deployment guide.
#
# Per the paper (Methods 2.4 and Fig 3 caption):
#   - the 4 real MaXim data sets (set1..set4) were each fit with the 8
#     architectures of Table 1 + the two "split" alpha variants
#     (n_exp_split, nn_ReLU_n_exp_split);
#   - the 3 artificial data sets (additive/antagonism/synergism) were each
#     fit with only the 5 architectures without splitting
#     (n, n_exp, nn_ReLU_n_exp, nn_ReLU_nn_ReLU_n_exp,
#     nn_ReLU_nn_ReLU_nn_ReLU_n_exp).
# That's 4*8 + 3*5 = 47 runs total.
#
# Every architecture's hidden-layer width is set to N_EXPOSURE (the number
# of compounds in the data set being fit), per Table 1's "n x n" weight
# matrices -- NOT a fixed width. That's why --hidden-layers is computed per
# data set below rather than hardcoded.
#
# Usage:
#   ./run/reproduce_paper.sh                     # all 47 runs, defaults
#   IMAGE=tktd-neuralodes:latest JOBS=8 ./run/reproduce_paper.sh
#   SCOPE=real ./run/reproduce_paper.sh           # only the 4 MaXim sets (32 runs)
#   SCOPE=artificial ./run/reproduce_paper.sh     # only the 3 artificial sets (15 runs)
#   N_UPDATE=200 N_ITER=200 N_ITER_WAIC=100 ./run/reproduce_paper.sh   # smoke test
#
# Requires: docker, GNU parallel (apt-get install -y parallel).
# Run from the repo root.
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-tktd-neuralodes:latest}"
JOBS="${JOBS:-$(nproc)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/output}"
SCOPE="${SCOPE:-all}"                # all | real | artificial
N_UPDATE="${N_UPDATE:-5000}"         # paper defaults (Methods 2.4)
N_ITER="${N_ITER:-2500}"
N_ITER_WAIC="${N_ITER_WAIC:-1500}"
N_CHAINS="${N_CHAINS:-3}"

if ! command -v parallel >/dev/null 2>&1; then
    echo "GNU parallel not found. Install it with: apt-get install -y parallel" >&2
    exit 1
fi
if [ ! -f "Dockerfile" ] || [ ! -d "src" ]; then
    echo "Run this script from the repo root (the one containing Dockerfile, R/, src/, data/)." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

## -- the 4 real MaXim data sets: id | csv path | comma-separated compound columns
declare -A DATASETS=(
    [1]="data/MaXim__raw_datasets__set1_CLEAN.csv|spiroxamine,prothioconazole,tebuconazole,trifloxystrobin,bixafen,fluopyram"
    [2]="data/MaXim__raw_datasets__set2_CLEAN.csv|spiromesifen,deltamethrin,triazophos,tralomethrin,flupyradifurone"
    [3]="data/MaXim__raw_datasets__set3_CLEAN.csv|thiacloprid,imidacloprid,cyfluthrin,clothianidin,beta-cyfluthrin,thiodicarb"
    [4]="data/MaXim__raw_datasets__set4_CLEAN.csv|flufenacet,diflufenican,metribuzin,flurtamone,aclonifen"
)
## -- the 3 artificial data sets
declare -A ARTIFICIAL=(
    [additive]="data/data_artificial_additive.csv|A,B"
    [antagonism]="data/data_artificial_antagonism.csv|A,B"
    [synergism]="data/data_artificial_synergism.csv|A,B"
)

## -- the paper's 8 architectures (Table 1 + the two split variants of
## Table 3), as --bridge generic flags:
##   layers : number of hidden layers, each of width N_EXPOSURE (0 = direct
##            linear bridge a + b.D_)
##   neg_slope, out_exp, alpha_split, alpha_out_exp, alpha_activation :
##            passed straight to run_TKTD_bayes.R --neg-slope/--out-exp/
##            --alpha-split/--alpha-out-exp/--alpha-activation
## alpha_activation=0 for nn_ReLU_n_exp_split reproduces a real asymmetry in
## the original Bayes_TKNNTD_split.R: ReLU on the D branch, no activation at
## all on the alpha branch, despite both sharing the same weights.
declare -A ARCH=(
    [n]="0:0:0:0:1:1"
    [n_exp]="0:0:1:0:1:1"
    [n_exp_split]="0:0:1:1:0:1"
    [nn_n_exp]="1:1:1:0:1:1"
    [nn_ReLU_n_exp]="1:0:1:0:1:1"
    [nn_ReLU_n_exp_split]="1:0:1:1:1:0"
    [nn_ReLU_nn_ReLU_n_exp]="2:0:1:0:1:1"
    [nn_ReLU_nn_ReLU_nn_ReLU_n_exp]="3:0:1:0:1:1"
)
## -- the 5 architectures used on the 3 artificial data sets (no splitting)
ARTIFICIAL_ARCHS=(n n_exp nn_ReLU_n_exp nn_ReLU_nn_ReLU_n_exp nn_ReLU_nn_ReLU_nn_ReLU_n_exp)

run_one() {
    local ID="$1" FILE="$2" MIX="$3" ARCH_NAME="$4"
    local SPEC="$5"
    IFS=':' read -r LAYERS NEG_SLOPE OUT_EXP ALPHA_SPLIT ALPHA_OUT_EXP ALPHA_ACT <<< "$SPEC"

    local N HIDDEN=""
    N=$(( $(grep -o ',' <<< "$MIX" | wc -l) + 1 ))
    for ((k = 0; k < LAYERS; k++)); do HIDDEN+="${N},"; done
    HIDDEN="${HIDDEN%,}"

    echo "[$(date '+%H:%M:%S')] START ${ID} / ${ARCH_NAME} (N=${N}, hidden-layers=[${HIDDEN}])"
    docker run --rm \
        -v "${OUTPUT_DIR}:/repo/output" \
        --add-host=host.docker.internal:host-gateway \
        ${MLFLOW_TRACKING_URI:+-e "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"} \
        "$IMAGE" \
        Rscript run/run_TKTD_bayes.R \
            --data "$FILE" --mixture "$MIX" --bridge generic \
            --hidden-layers "$HIDDEN" --neg-slope "$NEG_SLOPE" --out-exp "$OUT_EXP" \
            --alpha-split "$ALPHA_SPLIT" --alpha-out-exp "$ALPHA_OUT_EXP" --alpha-activation "$ALPHA_ACT" \
            --n-update "$N_UPDATE" --n-iter "$N_ITER" --n-iter-waic "$N_ITER_WAIC" --n-chains "$N_CHAINS" \
            --id "${ID}_${ARCH_NAME}" \
            --mlflow-experiment "TKTD-NeuralODEs-paper"
    echo "[$(date '+%H:%M:%S')] DONE  ${ID} / ${ARCH_NAME}"
}
export -f run_one
export OUTPUT_DIR IMAGE N_UPDATE N_ITER N_ITER_WAIC N_CHAINS MLFLOW_TRACKING_URI

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT

if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "real" ]; then
    for ID in "${!DATASETS[@]}"; do
        IFS='|' read -r FILE MIX <<< "${DATASETS[$ID]}"
        for ARCH_NAME in "${!ARCH[@]}"; do
            echo "${ID}|${FILE}|${MIX}|${ARCH_NAME}|${ARCH[$ARCH_NAME]}" >> "$jobs_file"
        done
    done
fi
if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "artificial" ]; then
    for ID in "${!ARTIFICIAL[@]}"; do
        IFS='|' read -r FILE MIX <<< "${ARTIFICIAL[$ID]}"
        for ARCH_NAME in "${ARTIFICIAL_ARCHS[@]}"; do
            echo "${ID}|${FILE}|${MIX}|${ARCH_NAME}|${ARCH[$ARCH_NAME]}" >> "$jobs_file"
        done
    done
fi

n_jobs=$(wc -l < "$jobs_file")
echo "Dispatching ${n_jobs} runs across ${JOBS} parallel workers (image: ${IMAGE})..."

parallel --will-cite -j "$JOBS" --colsep '\|' run_one {1} {2} {3} {4} {5} :::: "$jobs_file"

echo "All ${n_jobs} runs finished. Results in ${OUTPUT_DIR}/"
