# Cooling more while dissipating less

Code, numerical data, and figures for the Near-Equilibrium Anomalous Tradeoff
(NEAT): optimal cooling to a colder target can dissipate less work at the same
protocol duration in the slow-driving regime.

**Article:** [link to be added after publication]

This private working repository will evolve with the manuscript. The manuscript
is maintained separately in Overleaf; this repository contains the numerical
and figure-generation material.

## Reproduce

Use Python 3.11 and install the pinned plotting/numerical dependencies:

```sh
python -m pip install -r requirements.txt
python codes/qutrit_cooling.py
python codes/birth_death.py --mode plot
python codes/qutrit_endpoint_vectors.py
python codes/carnot_cycle_diagram.py
```

Outputs are written to `figures/` relative to this repository, independent of
the current working directory. The committed PDFs are the manuscript assets;
PNG and SVG versions are included for inspection and editing.

## Models and files

| Model | Code | Data and outputs |
| --- | --- | --- |
| Qutrit with constant relaxation eigenrates | `codes/qutrit_cooling.py` | `figures/qutrit-cooling/`: exact and numerical distances, paths, figures, validation |
| Birth–death ladder with structured and Ohmic baths | `codes/birth_death.py`, `codes/birth_death_plot.py` | `codes/data/birth-death/`: original curve samples and certification report; `figures/birth-death/`: combined plot |
| Endpoint tangent illustration | `codes/qutrit_endpoint_vectors.py` | `figures/qutrit-cooling/qutrit-endpoint-vectors.*` |
| Carnot-cycle schematic | `codes/carnot_cycle_diagram.py` | `figures/thermal-machines/` |

The qutrit calculation uses reference energies `(0, 0.003, 0.1)`, bath inverse
temperature `1`, and relaxation eigenrates `r_theta=0.5`, `r_phi=10`. Its
geodesic boundary-value solution is checked against the exact spherical
construction and Markov-generator identities.

The birth–death calculation uses energies `(0, 0.72, 2.40)` for `d=3` and
`(0, 0.72, 2.40, 2.70)` for `d=4`, with couplings `(1, 0.30)` and
`(1, 0.30, 1)`. The bath parameters are `A=10`, `Omega=1`, `gamma=0.30`,
`omega_c=10`; the Ohmic control sets only `A=0`. Bath inverse temperature is
`1`. Figure axes use `xi = beta_i / beta_f = T_f / T_i` and distance `D`.

## Reproducibility status

- **Qutrit:** fully regenerable without external input data. Use
  `python codes/qutrit_cooling.py --verify-only` for numerical checks alone.
- **Birth–death figure:** regenerated from the original CSV samples without
  smoothing, refitting, or changing the simulations. Its highlighted endpoints
  use the separate certification report. `plot-validation.json` records the
  input hash and checks on the displayed data.
- **Birth–death solver:** the original Fourier–Galerkin and finite-element
  routines and settings are retained. The original saved path-coefficient NPZ
  archives are currently missing. The `verify` and `recompute` modes therefore
  require those archives via `--seed-file`, plus JAX with 64-bit support.
  The archived convergence report records earlier checks; replotting does not
  rerun them. A tested JAX dependency pin should accompany restoration of the
  coefficient archives.
- `legacy/mpemba.nb` is the inherited Mathematica notebook, retained for
  provenance; it is not needed to reproduce the current figures.

## Updating the manuscript figures

Edit the scripts and data here, regenerate the affected outputs, inspect them,
and copy the corresponding files from `figures/` into Overleaf's `figures/`
directory. Commit the code and regenerated data/figures together so their
parameters and results remain traceable. Publication releases can be tagged
once the article and numerical material are finalized.
