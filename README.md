# Cooling more while dissipating less: anomalous tradeoffs from thermodynamic geometry

*Last updated October 26, 2026.*

**DISCLAIMER:** this repository is meant to be a technical supplement allowing the interested reader to easily reproduce all results and simulations from the paper [*Cooling more while dissipating less: anomalous tradeoffs from thermodynamic geometry* (arXiv 2610.27948)](https://arxiv.org/abs/2610.27948).
This repository is (mainly) AI generated, feeding as input some specifics, the original codes, data, and manuscript.
All material hereby presented has been reviewed by a human.

Code, numerical data, and figures for the paper


## Algorithms and numerical precision

| Calculation | Algorithm | Settings and interpretation |
| --- | --- | --- |
| Qutrit distance, Eq. (15), Fig. 2(a,b) | Exact great-circle distance in the rescaled spherical coordinates; independent integration of the geodesic boundary-value problem with SciPy `solve_bvp` | Float64; BVP residual tolerance `1e-9`, initial mesh 61 points, at most 20,000 nodes. Length integration: `quad` with absolute and relative tolerances `1e-11`. Validation checks 33 targets, including the two selected extrema. |
| Birth–death distance, Eqs. (18–20), Fig. 2(c) | Sine Fourier–Galerkin paths in log-gap coordinates, Gauss–Legendre quadrature, JAX automatic gradients in float64, SciPy L-BFGS-B; optimize energy then length, continue branches forwards/backwards and keep the shorter branch | `ftol=1e-14`; `gtol=1e-10` for energy and `1e-9` for length. Full-curve defaults: `(K,Q)=(18,384)` for d=3, `(16,320)` for d=4; Ohmic controls `(12,256)`. These stopping tolerances are not error bounds on the global geodesic distance. |
| Birth–death endpoint convergence checks | Higher-order Fourier calculation and a separate piecewise-linear path optimization | Archived high-order settings: `(K,Q)=(32,768)` for d=3 and `(24,640)` for d=4. Independent polylines: 90 segments × 12 quadrature nodes for d=3; 70 × 10 for d=4. |

**Observed precision, rather than just solver tolerances.** The archived
[birth–death convergence report](codes/data/birth-death/thermodynamic_length_certification.json)
compares targets `beta'/beta=4.25` and `9`. Raising the Fourier basis from
20 to 32 modes (d=3), or 18 to 24 (d=4), changes individual lengths by at most
`7.1e-6` and `1.5e-5`, respectively. For fixed high-order paths, raising quadrature
from 256 to 1024 nodes changes lengths by less than `5e-11`. This checks
quadrature only, not path optimization. Independent polylines change the
inversion gap by `1.14e-4` (d=3) and `6.36e-4` (d=4), compared with gaps of
`0.04322` and `0.04152`. These are empirical convergence checks; the report's
historical name “certification” does not mean a rigorous global error bound.

The qutrit verifier requires absolute length disagreement below `1e-7`,
maximum population-square-root path disagreement below `1e-6`, and sampled
Markov/metric residuals below `1e-8`; its measured errors are written to
[validation.json](figures/qutrit-cooling/validation.json). The 26 October 2026
run (Python 3.11.14 and the pinned dependencies) found maximum length error
`3.09e-14` and maximum path error `2.24e-12` over the 33 sampled targets.
These observed agreements are not universal error bounds.

**Reproducibility boundary:** the qutrit calculation runs from scratch, and
the birth–death panel can be replotted from the included CSV. The original
birth–death seed-coefficient NPZ archives are missing: `verify` and `recompute`
require an external `--seed-file` and a separately installed JAX with float64
support. The convergence figures above come from the archived report, not a
new solver run. The pinned requirements cover qutrit verification and plotting;
they do not include a validated JAX environment. All computations concern
slow-driving equilibrium-path geometry, not finite-time master-equation evolution.

## From paper notation to code

| Paper | Code |
| --- | --- |
| Qutrit relaxation rates `r_theta`, `r_phi` | `Model.r_theta=5`, `Model.r_phi=100` in `codes/qutrit_cooling.py` |
| Spectral amplitude `C` in Eq. (20) | `Bath.A=10` in `codes/birth_death.py`; the Ohmic control is `A=0` |
| `nu_0`, `gamma_B`, `nu_c` | `Bath.Omega=1`, `Bath.gamma=0.30`, `Bath.omega_c=10` |
| Cumulative populations `Omega_a` in Eq. (18) | `Pi` in the solver derivation; implementation transforms the diagonal cumulative-population metric to log-gap coordinates |
| Target ratio `beta'/beta` | `scale` for the qutrit; `lambda` for birth–death. Stored CSV coordinate `xi=T_final/T_initial` is its reciprocal. |

The positive-gap path is `y(s)=(1-s)y_initial+s*y_final+sum_k a_k sin(k*pi*s)`,
with `y_a=log(e_{a+1}-e_a)` and fixed endpoints. This enforces ordered levels
and samples the positive-gap control domain. The branch search is a numerical
minimization in this domain, not a proof of global optimality over all controls.

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
numerical reports. The current qutrit and birth–death PNG/SVG previews and validation reports were
regenerated on 26 October 2026. The qutrit paths now use the manuscript's energy
units `(0, 0.03, 1)` and `beta=0.1`; older reports used an equivalent rescaling.
The standalone `qutrit-endpoint-vectors.*` and `endpoint-vectors.json` remain
legacy illustrations, not the current sphere panel; its vector data are in
`sphere-endpoint-vectors.json`.

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
`(1, 0.30, 1)`. In manuscript notation the bath parameters are `C=10` (code: `A=10`),
`nu_0=1`, `gamma_B=0.30`, `nu_c=10`; the Ohmic control sets only `C=0` (code: `A=0`).
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
