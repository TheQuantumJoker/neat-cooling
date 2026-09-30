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
python codes/carnot_cycle_diagram.py
```

Outputs are written to `figures/` relative to this repository, independent of
the current working directory. The committed PDFs are the current manuscript
assets. The introductory qualitative illustration is supplied as a PNG.
The qutrit and birth–death scripts additionally generate PNG/SVG previews and
numerical reports. Existing PNG/SVG previews, qutrit CSV files, and generated
JSON reports are retained as **earlier snapshots** and have not been refreshed
in this synchronization. Use the committed PDFs for the current manuscript;
rerun the scripts to regenerate the other formats and reports with current parameters.

## Models and files

| Model | Code | Data and outputs |
| --- | --- | --- |
| Qutrit with constant relaxation eigenrates | `codes/qutrit_cooling.py` | `figures/qutrit-cooling/`: sphere geometry with endpoint vectors and exact-distance plot; running the script also generates paths and validation data |
| Birth–death ladder with structured and Ohmic baths | `codes/birth_death.py`, `codes/birth_death_plot.py` | `codes/data/birth-death/`: original curve samples and certification report; `figures/birth-death/`: combined plot |
| Carnot-cycle schematic | `codes/carnot_cycle_diagram.py` | `figures/thermal-machines/` |
| Introductory qualitative illustration | Supplied artwork | `figures/introduction/neat-vs-mpemba-final.png` |

The qutrit calculation uses reference energies `(0, 0.03, 1)`, bath inverse
temperature `0.1`, and relaxation eigenrates `r_theta=5`, `r_phi=100`. Its
geodesic boundary-value solution is checked against the exact spherical
construction and Markov-generator identities.

The birth–death calculation uses energies `(0, 0.72, 2.40)` for `d=3` and
`(0, 0.72, 2.40, 2.70)` for `d=4`, with couplings `(1, 0.30)` and
`(1, 0.30, 1)`. In manuscript notation the bath parameters are `A=10`,
`nu_0=1`, `gamma_B=0.30`, `nu_c=10`; the Ohmic control sets only `A=0`.
Bath inverse temperature is `1`. Both distance panels use the logarithmic
horizontal coordinate `beta'/beta`, increasing toward colder targets, and
vertical coordinate `L(beta -> beta')`. The preserved birth–death CSV uses
the reciprocal temperature ratio internally; the plotting code converts it.

The manuscript combines the qutrit sphere, qutrit distance, and birth–death
distance into three panels in one row. The qutrit sphere uses Gibbs-population
labels `sqrt(omega_a)` and includes the arriving tangent and Gibbs-curve
derivative. The distance panel displays the exact curve without numerical
geodesic markers. The Carnot schematic has enlarged labels for its end-matter
placement. The introductory figure is supplied artwork, not a simulation or
a Python-generated plot.

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
- `codes/qutrit_endpoint_vectors.py` is the older standalone tangent illustration;
  the current sphere panel incorporates its vectors directly in `qutrit_cooling.py`.
- `legacy/mpemba.nb` is the inherited Mathematica notebook, retained for
  provenance; it is not needed to reproduce the current figures.

## Updating the manuscript figures

Edit the scripts and data here, regenerate the affected outputs, inspect them,
and copy the corresponding files from `figures/` into Overleaf's `figures/`
directory. Commit the code and regenerated data/figures together so their
parameters and results remain traceable. Publication releases can be tagged
once the article and numerical material are finalized.

The current repository snapshot was synchronized from the manuscript's active
code and figure assets on 30 September 2026. No numerical solvers were rerun
as part of that synchronization; the birth–death source data and certification
report are unchanged.
