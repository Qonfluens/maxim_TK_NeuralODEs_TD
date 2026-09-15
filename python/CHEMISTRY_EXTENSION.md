# Adding compound properties to the TK-NN-TD model

How to feed molecular information (logP, Kow, Koa, SMILES, and ADME-type
properties: accessibility, availability, metabolisation, excretion) into the
network that sits between the kinetics and the damage, keeping the spirit of
the paper.

This is a design document, not yet implemented code. It is written to be built
in stages, each one independently testable against the reproduction baseline.

---

## 1. Why the current model cannot use chemistry at all

In the model as published, every per-compound quantity is a **free scalar
fitted independently for each data set**:

| quantity | shape | what it means |
|---|---|---|
| `kd[j]` | one per compound | dominant rate constant of compound *j* |
| `alpha[j]` | one per compound (split mode) | tolerance threshold for compound *j* |
| `W[l]` | `n x n` per data set | how compounds interact on the way to damage |

Three consequences follow, and they are the reason the extension is worth doing:

1. **Nothing transfers.** `kd` for tebuconazole in set 1 is a different, unrelated
   parameter from `kd` for tebuconazole anywhere else. Fitting more data sets
   does not sharpen any estimate.
2. **No prediction for new compounds.** A compound absent from the training data
   has no `kd`, no `alpha`, and no row or column in `W`. The model cannot say
   anything about it.
3. **Interactions are not learnable in any general sense.** `W` is indexed by
   *position in this data set's compound list*. A synergy learned between
   positions 2 and 5 of set 1 carries no meaning in set 3.

The fix is the same in all three cases: stop treating per-compound quantities as
free parameters and start **predicting them from the molecule**.

```
        free scalar per compound                predicted from structure
   kd[j]        <-- fitted          ===>    kd[j] = f_theta(descriptors[j])
   alpha[j]     <-- fitted          ===>    alpha[j] = g_theta(descriptors[j])
   W[k,j]       <-- fitted          ===>    W[k,j] = h_theta(e[k], e[j])
```

`f`, `g`, `h` are small networks **shared across every compound and every data
set**. The number of parameters then stops growing with the number of compounds,
and it becomes possible to train on all four MaXim sets at once.

---

## 2. Which properties go where

Not every descriptor belongs in the same place. Mapping them onto the parts of
the model they physically drive keeps the model interpretable and is a much
stronger prior than feeding everything into one big network.

| property | what it governs | where it enters |
|---|---|---|
| logP / logKow | partitioning into lipid tissue | `kd`, and the uptake term |
| Koa | air-organism partitioning (relevant for vapour exposure) | uptake term |
| molecular weight, TPSA | membrane permeability | uptake term |
| accessibility, availability | fraction of nominal exposure actually reaching the organism | scales `X` before the TK step |
| metabolisation rate | loss of parent compound | elimination term |
| excretion rate | loss from the organism | elimination term |
| mode of action / target class | which biological pathway is hit | compound embedding, and interactions |
| SMILES / molecular graph | everything not captured above | learned embedding |

Two of these deserve emphasis:

- **Accessibility and availability multiply the exposure, they do not change the
  kinetics.** They belong at `X`, not inside `kd`. `X_effective[j] = a(desc_j) * X[j]`
  with `a` in `(0, 1)`. Note that this is only identifiable if `a` varies across
  compounds and the bridge's first-layer weights are constrained, otherwise it is
  absorbed into `W`. Start with `a` fixed from measured values rather than fitted.
- **Metabolisation and excretion are distinct elimination routes** that the
  current one-compartment TK collapses into the single constant `kd`. Splitting
  them requires a real ODE (section 4).

---

## 3. Stage 1 — chemistry-conditioned parameters (recommended starting point)

Keep the model structure exactly as reproduced, and replace only the
per-compound scalars.

```python
class CompoundEncoder(nn.Module):
    """descriptors -> embedding, shared by every compound in every data set."""
    def __init__(self, n_desc, d_emb=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_desc, 32), nn.SiLU(), nn.Linear(32, d_emb)
        )

    def forward(self, desc):        # (n_compounds, n_desc)
        return self.net(desc)       # (n_compounds, d_emb)


class ChemistryHeads(nn.Module):
    """embedding -> the quantities that used to be free scalars."""
    def __init__(self, d_emb=16):
        super().__init__()
        self.kd_head = nn.Linear(d_emb, 1)
        self.alpha_head = nn.Linear(d_emb, 1)

    def forward(self, e):
        # still on the log10 scale, so the paper's priors remain meaningful
        return self.kd_head(e).squeeze(-1), self.alpha_head(e).squeeze(-1)
```

The descriptor vector should be standardised across the training compounds, and
should include the measured physico-chemical values first (logP, logKow, logKoa,
MW, TPSA) before any learned structural representation.

**Keep a residual.** Forcing `kd` to be an exact function of descriptors is too
rigid for a first attempt; real compounds deviate. Use

```
kd_log10[j] = kd_head(e_j) + delta_j,      delta_j ~ N(0, sigma_delta)
```

with `sigma_delta` small and either fixed or fitted. With `sigma_delta` large the
model falls back to the published one (`delta` absorbs everything); with
`sigma_delta = 0` it is fully chemistry-driven. That single knob makes the
extension **continuously comparable to the baseline**, which is exactly what you
want when checking you have not broken anything.

### What this buys immediately

- Train jointly on all 4 MaXim data sets: 22 compounds instead of 5-6.
- Leave-one-compound-out validation becomes possible, and is the honest test of
  whether chemistry actually predicts kinetics.

---

## 4. Stage 2 — a real ODE for ADME (where "Neural ODE" becomes literal)

The published TK is a one-compartment model with an analytical solution, which
is why the JAGS code can write `X * (1 - exp(-kd t))` directly. Metabolisation
and excretion cannot be separated inside that single `kd`. Replace it with an
explicit system and integrate it:

```
dC_j/dt  = k_in(desc_j) * a(desc_j) * X_j  -  k_out(desc_j) * C_j  -  k_met(desc_j) * C_j
dM_j/dt  = k_met(desc_j) * C_j             -  k_exc(desc_j) * M_j
```

`C` is the internal parent concentration, `M` the metabolite. Damage is then
driven by both: the bridge takes `[C_1..C_n, M_1..M_n]` instead of `[D_1..D_n]`,
which lets the model represent a compound whose metabolite is the toxic species
(a real and common case).

Integrate with `torchdiffeq.odeint_adjoint`, which backpropagates through the
solver in constant memory:

```python
from torchdiffeq import odeint_adjoint as odeint
state = odeint(tk_rhs, state0, t_eval, method="dopri5", rtol=1e-6, atol=1e-8)
```

**Do not start here.** Validate Stage 1 first: with `k_met = k_exc = 0` the ODE
must reproduce the analytical solution to solver tolerance, and that is the test
to write before trusting any of it. The analytical path should stay in the code
as the default, because it is faster and exactly matches the published model.

---

## 5. Stage 3 — interactions predicted from chemistry

This is the scientifically interesting part and the real payoff.

Currently `W[k,j]` is a free `n x n` matrix, so interactions are memorised per
data set. Make the interaction a **function of the two molecules involved**:

```python
class InteractionBridge(nn.Module):
    """W[k,j] = h(e_k, e_j): the interaction depends on the pair, not the index."""
    def __init__(self, d_emb=16):
        super().__init__()
        self.pair = nn.Sequential(
            nn.Linear(2 * d_emb, 32), nn.SiLU(), nn.Linear(32, 1)
        )

    def forward(self, e):                       # (n, d_emb)
        n = e.shape[0]
        pairs = torch.cat([
            e.unsqueeze(1).expand(n, n, -1),
            e.unsqueeze(0).expand(n, n, -1),
        ], dim=-1)                              # (n, n, 2*d_emb)
        return self.pair(pairs).squeeze(-1)     # (n, n)
```

Three properties make this a strict improvement over a free matrix:

1. **Permutation equivariance.** Reordering the compound list reorders `W`
   consistently. The model no longer depends on column order in the CSV.
2. **Variable `n`.** The same trained model applies to a 5-compound and a
   6-compound data set. Joint training across all four sets becomes possible.
3. **Generalisation to unseen pairs.** A pair never observed together gets a
   predicted interaction from the two molecules' properties.

An attention layer over the compound set is the natural generalisation (each
compound attends to the others, weighted by chemical similarity), and is worth
trying once the bilinear form works.

**Testable prediction, and how to falsify it:** train on the 4 MaXim sets
holding out one compound entirely, then predict its mixtures. If chemistry-driven
interactions are real, held-out performance beats a baseline that assumes strict
additivity. If it does not, the descriptors are not carrying interaction
information and the honest conclusion is that this data cannot support the claim.
The 3 artificial data sets (additive / antagonism / synergism, where ground
truth is known by construction) are the right place to check that the machinery
can recover a known interaction before trusting it on real data.

---

## 6. Where to get the descriptors

```python
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

mol = Chem.MolFromSmiles(smiles)
features = {
    "logP":  Crippen.MolLogP(mol),        # computed logP
    "MW":    Descriptors.MolWt(mol),
    "TPSA":  rdMolDescriptors.CalcTPSA(mol),
    "HBD":   rdMolDescriptors.CalcNumHBD(mol),
    "HBA":   rdMolDescriptors.CalcNumHBA(mol),
    "rotB":  rdMolDescriptors.CalcNumRotatableBonds(mol),
}
```

Guidance on sourcing, in order of preference:

1. **Measured values where they exist.** For registered pesticides, the PPDB and
   regulatory dossiers give measured logP, water solubility, DT50, and Koc.
   These beat computed values and should be used first.
2. **RDKit descriptors** for what is missing. Cheap, deterministic, no training.
3. **Morgan/ECFP fingerprints** if you want structural information beyond the
   descriptor list. 2048-bit folded to a small embedding by a linear layer.
4. **A graph neural network on the molecular graph** only once stages 1-3 work.
   With ~22 compounds, a GNN trained end-to-end will overfit badly; it only
   becomes reasonable with a pretrained encoder frozen or lightly fine-tuned.

The practical constraint is worth stating plainly: **22 compounds is a very small
training set for anything chemistry-driven.** That argues for few descriptors,
strong regularisation, measured values over learned ones, and leave-one-compound-out
validation on every claim. A GNN is the wrong tool at this sample size.

---

## 7. Suggested file layout

```
python/tktd/
    chemistry/
        descriptors.py    # SMILES -> descriptor table, caching, standardisation
        encoder.py        # CompoundEncoder, ChemistryHeads
        interaction.py    # InteractionBridge
    ode.py                # torchdiffeq TK, with the analytical path as default
    multi_dataset.py      # joint training across data sets with shared encoders
data/
    compounds.csv         # name, SMILES, logP, logKow, logKoa, source, ...
```

`data/compounds.csv` is the first thing to create, and the only step that cannot
be automated: one row per compound across all four MaXim sets, with a `source`
column recording where each number came from. Everything else is downstream of it.

---

## 8. Order of work

| step | deliverable | how you know it worked |
|---|---|---|
| 0 | reproduce the 47 runs in PyTorch | done: see `README.md`, deviance matches JAGS exactly |
| 1 | `data/compounds.csv` for all 22 compounds | every compound has a SMILES and a sourced logP |
| 2 | `kd` predicted from descriptors, with residual | with large `sigma_delta`, deviance matches step 0 |
| 3 | joint training over the 4 real data sets | leave-one-compound-out beats a per-data-set fit |
| 4 | chemistry-driven interaction matrix | recovers the known interaction on the artificial sets |
| 5 | ODE-based ADME | reduces to the analytical solution when `k_met = k_exc = 0` |

Each step has a check that fails loudly if the step broke something, and each is
worth stopping at if the result is negative. Step 2's fallback property (large
`sigma_delta` reproduces the baseline) is the single most useful safeguard in
the list: it means the extension can always be compared against the published
model on equal terms.
