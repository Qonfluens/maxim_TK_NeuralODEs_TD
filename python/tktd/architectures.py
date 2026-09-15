"""The paper's model and data set tables, shared by the CLI and the driver.

Transcribed from run/reproduce_paper.sh so the PyTorch runs cover exactly the
same 47 fits: the 4 real MaXim data sets x 8 architectures (Table 1 plus the
two 'split' variants of Table 3), and the 3 artificial data sets x the 5
architectures without splitting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    """One row of the paper's model table.

    n_hidden is a count of hidden layers; each one has width n_X (the number of
    compounds in the data set being fit), following Table 1's 'n x n' weight
    matrices. The width is therefore resolved per data set, never hardcoded.
    """

    name: str
    n_hidden: int
    neg_slope: float
    out_exp: int
    alpha_split: int
    alpha_out_exp: int
    alpha_activation: int

    def hidden_layers(self, n_X: int) -> tuple[int, ...]:
        return tuple([n_X] * self.n_hidden)


# alpha_activation = 0 on nn_ReLU_n_exp_split reproduces a real asymmetry in
# the original Bayes_TKNNTD_split.R: ReLU on the damage branch, no activation
# at all on the alpha branch, despite both sharing the same weights.
ARCHITECTURES: dict[str, Architecture] = {
    a.name: a for a in [
        Architecture("n",                             0, 0.0, 0, 0, 1, 1),
        Architecture("n_exp",                         0, 0.0, 1, 0, 1, 1),
        Architecture("n_exp_split",                   0, 0.0, 1, 1, 0, 1),
        Architecture("nn_n_exp",                      1, 1.0, 1, 0, 1, 1),
        Architecture("nn_ReLU_n_exp",                 1, 0.0, 1, 0, 1, 1),
        Architecture("nn_ReLU_n_exp_split",           1, 0.0, 1, 1, 1, 0),
        Architecture("nn_ReLU_nn_ReLU_n_exp",         2, 0.0, 1, 0, 1, 1),
        Architecture("nn_ReLU_nn_ReLU_nn_ReLU_n_exp", 3, 0.0, 1, 0, 1, 1),
    ]
}

# architectures applied to the artificial data sets (no alpha splitting)
ARTIFICIAL_ARCHS = (
    "n", "n_exp", "nn_ReLU_n_exp",
    "nn_ReLU_nn_ReLU_n_exp", "nn_ReLU_nn_ReLU_nn_ReLU_n_exp",
)

REAL_DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "1": ("data/MaXim__raw_datasets__set1_CLEAN.csv",
          ("spiroxamine", "prothioconazole", "tebuconazole",
           "trifloxystrobin", "bixafen", "fluopyram")),
    "2": ("data/MaXim__raw_datasets__set2_CLEAN.csv",
          ("spiromesifen", "deltamethrin", "triazophos",
           "tralomethrin", "flupyradifurone")),
    "3": ("data/MaXim__raw_datasets__set3_CLEAN.csv",
          ("thiacloprid", "imidacloprid", "cyfluthrin",
           "clothianidin", "beta-cyfluthrin", "thiodicarb")),
    "4": ("data/MaXim__raw_datasets__set4_CLEAN.csv",
          ("flufenacet", "diflufenican", "metribuzin",
           "flurtamone", "aclonifen")),
}

ARTIFICIAL_DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "additive": ("data/data_artificial_additive.csv", ("A", "B")),
    "antagonism": ("data/data_artificial_antagonism.csv", ("A", "B")),
    "synergism": ("data/data_artificial_synergism.csv", ("A", "B")),
}


def all_jobs(scope: str = "all") -> list[tuple[str, str, tuple[str, ...], Architecture]]:
    """(dataset_id, csv_path, compound_columns, architecture) for every run."""
    jobs = []
    if scope in ("all", "real"):
        for ds_id, (path, cols) in REAL_DATASETS.items():
            for arch in ARCHITECTURES.values():
                jobs.append((ds_id, path, cols, arch))
    if scope in ("all", "artificial"):
        for ds_id, (path, cols) in ARTIFICIAL_DATASETS.items():
            for name in ARTIFICIAL_ARCHS:
                jobs.append((ds_id, path, cols, ARCHITECTURES[name]))
    return jobs
