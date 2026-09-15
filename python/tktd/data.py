"""CSV loading and preprocessing for the TKTD-IT model.

Ported 1:1 from .prepare_TKTD_data() in R/Bayes_TK-NN-TD_generic.R so that the
PyTorch model is fed exactly the same arrays as the JAGS model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Columns that are never exposure substances/routes. Mirrors
# NON_MIXTURE_COLUMNS in run/run_TKTD_bayes.R, plus the two spellings pandas
# gives R's unnamed row-name column.
NON_MIXTURE_COLUMNS = frozenset({
    "X", "Unnamed: 0", "", "time", "file", "replicate", "Nsurv", "Nprec",
    "unit_ai", "M_number", "Mixture_abcSorted", "formulation",
    "Psurv", "Interaction", "i_row", "lag_i_row", "i_prec",
})


@dataclass
class TKTDDataset:
    """Preprocessed survival data, in the exact shape the JAGS model expects.

    All index arrays are 0-based (JAGS/R used 1-based).
    """

    time: np.ndarray          # (n_data,) float
    X: np.ndarray             # (n_data, n_X) float, exposure concentrations
    Nsurv: np.ndarray         # (n_data,) float
    Nprec: np.ndarray         # (n_data,) float
    i_prec: np.ndarray        # (n_data,) int, row index of the previous time point
    replicate_ID: np.ndarray  # (n_data,) int
    time_ID: np.ndarray       # (n_data,) int, position within the replicate
    compounds: list[str] = field(default_factory=list)
    replicates: list[str] = field(default_factory=list)

    @property
    def n_data(self) -> int:
        return len(self.time)

    @property
    def n_X(self) -> int:
        return self.X.shape[1]

    @property
    def n_replicates(self) -> int:
        return len(self.replicates)

    @property
    def n_time_max(self) -> int:
        return int(self.time_ID.max()) + 1


def infer_mixture_columns(d: pd.DataFrame) -> list[str]:
    """Any numeric column that is not in NON_MIXTURE_COLUMNS is an exposure."""
    candidates = [c for c in d.columns if c not in NON_MIXTURE_COLUMNS]
    mixture = [c for c in candidates if pd.api.types.is_numeric_dtype(d[c])]
    if not mixture:
        raise ValueError(
            "No exposure column could be auto-detected. Available columns: "
            f"{', '.join(map(str, d.columns))}. Pass --mixture col1,col2,... explicitly."
        )
    return mixture


def load_tktd_csv(
    path: str,
    mixture_cols: list[str] | None = None,
    sep: str = ",",
) -> TKTDDataset:
    d = pd.read_csv(path, sep=sep)

    if mixture_cols is None:
        mixture_cols = infer_mixture_columns(d)
    else:
        missing = [c for c in mixture_cols if c not in d.columns]
        if missing:
            raise ValueError(
                f"Mixture columns not found in {path}: {', '.join(missing)}. "
                f"Available columns: {', '.join(map(str, d.columns))}"
            )

    for required in ("time", "replicate", "Nsurv"):
        if required not in d.columns:
            raise ValueError(f"Column '{required}' is required but missing from {path}.")

    rep = d["replicate"].astype(str)

    # Same precondition as the R engine: each replicate must occupy one
    # contiguous block of rows, first row at time == 0.
    n_blocks = int((rep != rep.shift(1)).sum())
    if n_blocks != rep.nunique():
        warnings.warn(
            "Rows are not grouped into contiguous blocks by 'replicate'. "
            "Nprec/i_prec are computed from row order and assume the data is "
            "sorted by replicate then by time.",
            stacklevel=2,
        )

    replicates = list(pd.unique(rep))
    rep_to_id = {r: i for i, r in enumerate(replicates)}
    replicate_ID = rep.map(rep_to_id).to_numpy(dtype=np.int64)
    time_ID = d.groupby(rep, sort=False).cumcount().to_numpy(dtype=np.int64)

    # Nprec: within each replicate, the survivor count at the previous time
    # point; the first row of a replicate uses its own count (R: c(a[1], a[-n])).
    Nsurv = d["Nsurv"].to_numpy(dtype=np.float64)
    Nprec = pd.Series(Nsurv).groupby(replicate_ID, sort=False).shift(1)
    Nprec = Nprec.fillna(pd.Series(Nsurv)).to_numpy(dtype=np.float64)

    # i_prec: the previous row overall, except at time == 0 where a replicate
    # starts and the row points at itself.
    time = d["time"].to_numpy(dtype=np.float64)
    i_row = np.arange(len(d), dtype=np.int64)
    lag_i_row = np.concatenate([[0], i_row[:-1]])
    i_prec = np.where(time == 0, i_row, lag_i_row)

    return TKTDDataset(
        time=time,
        X=d[mixture_cols].to_numpy(dtype=np.float64),
        Nsurv=Nsurv,
        Nprec=Nprec,
        i_prec=i_prec,
        replicate_ID=replicate_ID,
        time_ID=time_ID,
        compounds=list(mixture_cols),
        replicates=replicates,
    )
