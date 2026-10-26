#!/usr/bin/env python3
"""Birth–death numerical methods for the NEAT cooling paper.

The model is a nearest-neighbour d-level Davies ladder.  In cumulative
population coordinates Pi_m = sum_{i=0}^m p_i, the thermodynamic metric is

    d ell^2 = beta^{-1} sum_m dPi_m^2 / w_m,
    w_m = |c_m|^2 J(Delta_m) p_m p_{m+1}/(p_m-p_{m+1}).

Positive gaps are parameterized by y_m = log Delta_m.  Geodesics are represented
by a Fourier-Galerkin path and optimized with analytic JAX gradients.  The script
has three modes:

    plot       regenerate figures from the included converged curve data;
    verify     re-evaluate and locally reoptimize the certified inversion pair;
    recompute  regenerate all curves by forward/backward branch continuation.

Units: hbar = k_B = beta_bath = T_initial = 1.  The endpoint parameter lambda
satisfies lambda = T_initial/T_final, so T_final/T_initial = 1/lambda.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize


@dataclass(frozen=True)
class Bath:
    A: float = 10.0  # Spectral amplitude C in Eq. (20) of the manuscript.
    gamma: float = 0.30
    Omega: float = 1.0
    omega_c: float = 10.0


@dataclass(frozen=True)
class SystemSpec:
    name: str
    gaps: tuple[float, ...]
    couplings: tuple[float, ...]

    @property
    def d(self) -> int:
        return len(self.gaps) + 1

    @property
    def energies(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(np.asarray(self.gaps, dtype=float))]


BATH = Bath()
OHMIC_CONTROL = Bath(A=0.0, gamma=BATH.gamma, Omega=BATH.Omega, omega_c=BATH.omega_c)
D3 = SystemSpec("d3", (0.72, 1.68), (1.0, 0.30))
D4 = SystemSpec("d4", (0.72, 1.68, 0.30), (1.0, 0.30, 1.0))
LAMBDA_NEAR = 4.25
LAMBDA_FAR = 9.0


def structured_spectrum_numpy(omega: np.ndarray | float, bath: Bath = BATH) -> np.ndarray:
    """Ohmic exponential background plus an underdamped Brownian mode.

    A is the resonant-mode/background ratio at omega=Omega.  Consequently the
    total structured/background ratio at resonance is 1+A (=11 here).
    """
    w = np.asarray(omega, dtype=float)
    den = (bath.Omega**2 - w**2) ** 2 + (bath.gamma * w) ** 2
    background = w * np.exp(-w / bath.omega_c)
    mode = (
        bath.A
        * np.exp(-bath.Omega / bath.omega_c)
        * bath.gamma**2
        * bath.Omega**2
        * w
        / den
    )
    return background + mode


def gibbs_populations(gaps: Sequence[float], beta: float = 1.0) -> np.ndarray:
    energies = np.r_[0.0, np.cumsum(np.asarray(gaps, dtype=float))]
    z = -beta * energies
    z -= np.max(z)
    p = np.exp(z)
    return p / np.sum(p)


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "JAX is required for verify/recompute in addition to requirements.txt; see README.md."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _resize_coefficients(x: np.ndarray, n: int, k_old: int, k_new: int) -> np.ndarray:
    old = np.asarray(x, dtype=float).reshape(n, k_old)
    new = np.zeros((n, k_new), dtype=float)
    new[:, : min(k_old, k_new)] = old[:, : min(k_old, k_new)]
    return new.ravel()


def _evaluate_fourier_path(
    x: np.ndarray,
    n: int,
    k_modes: int,
    y0: np.ndarray,
    y1: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float).reshape(n, k_modes)
    modes = np.arange(1, k_modes + 1, dtype=float)
    s = np.asarray(s, dtype=float)
    y = (
        y0[None, :]
        + s[:, None] * (y1 - y0)[None, :]
        + np.sin(np.pi * s[:, None] * modes[None, :]) @ a.T
    )
    ydot = (
        (y1 - y0)[None, :]
        + (np.cos(np.pi * s[:, None] * modes[None, :]) * (np.pi * modes)[None, :])
        @ a.T
    )
    return y, ydot


def make_fourier_problem(
    system: SystemSpec,
    bath: Bath,
    k_modes: int,
    quadrature_nodes: int,
):
    """Return JIT-compiled metric, energy and length functions with gradients."""
    jax, jnp = _import_jax()
    n = system.d - 1
    couplings = jnp.asarray(system.couplings, dtype=jnp.float64)
    params = jnp.asarray([bath.A, bath.gamma, bath.Omega, bath.omega_c], dtype=jnp.float64)

    gl_x, gl_w = leggauss(quadrature_nodes)
    s = jnp.asarray((gl_x + 1.0) / 2.0)
    weights = jnp.asarray(gl_w / 2.0)
    modes = jnp.arange(1, k_modes + 1, dtype=jnp.float64)
    sin_matrix = jnp.sin(jnp.pi * s[:, None] * modes[None, :])
    cos_matrix = (jnp.pi * modes[None, :]) * jnp.cos(jnp.pi * s[:, None] * modes[None, :])

    def spectrum(w):
        A, gamma, Omega, omega_c = params
        den = (Omega**2 - w**2) ** 2 + (gamma * w) ** 2
        return w * jnp.exp(-w / omega_c) + (
            A * jnp.exp(-Omega / omega_c) * gamma**2 * Omega**2 * w / den
        )

    def metric_batch(y):
        gaps = jnp.exp(y)
        energies = jnp.concatenate(
            [jnp.zeros((y.shape[0], 1), dtype=y.dtype), jnp.cumsum(gaps, axis=1)],
            axis=1,
        )
        logits = -energies
        logits -= jnp.max(logits, axis=1, keepdims=True)
        p = jnp.exp(logits)
        p /= jnp.sum(p, axis=1, keepdims=True)
        cumulative = jnp.cumsum(p, axis=1)[:, :-1]

        m_index = jnp.arange(n)[:, None]
        a_index = jnp.arange(n)[None, :]
        q_m = cumulative[:, :, None]
        q_a = cumulative[:, None, :]
        gap_a = gaps[:, None, :]
        jacobian = gap_a * jnp.where(
            m_index[None, :, :] <= a_index[None, :, :],
            q_m * (1.0 - q_a),
            q_a * (1.0 - q_m),
        )

        # Stable form of p_m p_{m+1}/(p_m-p_{m+1}) = p_m/expm1(Delta_m).
        conductance = (
            couplings[None, :] ** 2
            * spectrum(gaps)
            * p[:, :-1]
            / jnp.expm1(gaps)
        )
        return jnp.einsum(
            "bma,bm,bmc->bac", jacobian, 1.0 / conductance, jacobian
        )

    def speed_squared(x, y0, y1):
        coefficients = x.reshape((n, k_modes))
        y = y0[None, :] + s[:, None] * (y1 - y0)[None, :] + sin_matrix @ coefficients.T
        ydot = (y1 - y0)[None, :] + cos_matrix @ coefficients.T
        metric = metric_batch(y)
        return jnp.einsum("bi,bij,bj->b", ydot, metric, ydot)

    def energy(x, y0, y1):
        return 0.5 * jnp.sum(weights * speed_squared(x, y0, y1))

    def length(x, y0, y1):
        return jnp.sum(weights * jnp.sqrt(jnp.maximum(speed_squared(x, y0, y1), 1e-300)))

    energy_vg = jax.jit(jax.value_and_grad(energy))
    length_vg = jax.jit(jax.value_and_grad(length))
    length_jit = jax.jit(length)
    metric_jit = jax.jit(metric_batch)
    return energy_vg, length_vg, length_jit, metric_jit


def _scipy_minimize_jax(
    value_and_grad,
    x0: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    maxiter: int,
    gtol: float = 1e-9,
):
    _, jnp = _import_jax()
    y0j = jnp.asarray(y0)
    y1j = jnp.asarray(y1)

    def fun(x):
        value, grad = value_and_grad(jnp.asarray(x), y0j, y1j)
        return float(value), np.asarray(grad, dtype=float)

    return minimize(
        fun,
        np.asarray(x0, dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={
            "ftol": 1e-14,
            "gtol": gtol,
            "maxiter": maxiter,
            "maxls": 60,
            "maxcor": 40,
        },
    )


def _reparameterize_coefficients(
    x: np.ndarray,
    system: SystemSpec,
    bath: Bath,
    k_modes: int,
    y0: np.ndarray,
    y1: np.ndarray,
    metric_batch,
    sample_count: int = 5000,
) -> np.ndarray:
    """Refit the same geometric path at approximately constant metric speed."""
    _, jnp = _import_jax()
    n = system.d - 1
    s = np.linspace(0.0, 1.0, sample_count)
    y, ydot = _evaluate_fourier_path(x, n, k_modes, y0, y1, s)
    metric = np.asarray(metric_batch(jnp.asarray(y)))
    speed = np.sqrt(np.maximum(np.einsum("bi,bij,bj->b", ydot, metric, ydot), 1e-300))
    arc = np.r_[0.0, cumulative_trapezoid(speed, s)]
    arc /= arc[-1]
    keep = np.r_[True, np.diff(arc) > 1e-12]
    arc = arc[keep]
    y = y[keep]

    u = np.linspace(0.0, 1.0, max(1000, 35 * k_modes))
    y_uniform = np.stack([np.interp(u, arc, y[:, a]) for a in range(n)], axis=1)
    residual = y_uniform - (y0[None, :] + u[:, None] * (y1 - y0)[None, :])
    sine = np.sin(np.pi * u[:, None] * np.arange(1, k_modes + 1)[None, :])
    regularizer = 1e-12 * np.arange(1, k_modes + 1, dtype=float) ** 4
    normal = sine.T @ sine + np.diag(regularizer)
    coefficients = np.stack(
        [np.linalg.solve(normal, sine.T @ residual[:, a]) for a in range(n)], axis=0
    )
    return coefficients.ravel()


def optimize_fourier_path(
    system: SystemSpec,
    bath: Bath,
    lam: float,
    k_modes: int,
    quadrature_nodes: int,
    x0: np.ndarray,
    energy_iterations: int,
    length_iterations: int,
    reparameterization_cycles: int = 1,
):
    n = system.d - 1
    gaps = np.asarray(system.gaps, dtype=float)
    y0 = np.log(gaps)
    y1 = np.log(lam * gaps)
    energy_vg, length_vg, length_jit, metric_batch = make_fourier_problem(
        system, bath, k_modes, quadrature_nodes
    )

    energy_result = _scipy_minimize_jax(
        energy_vg, x0, y0, y1, maxiter=energy_iterations, gtol=1e-10
    )
    current = energy_result.x
    best_result = None
    for _ in range(reparameterization_cycles + 1):
        result = _scipy_minimize_jax(
            length_vg, current, y0, y1, maxiter=length_iterations, gtol=1e-9
        )
        if best_result is None:
            best_result = result
        else:
            _, jnp = _import_jax()
            old_value = float(length_jit(jnp.asarray(best_result.x), jnp.asarray(y0), jnp.asarray(y1)))
            new_value = float(length_jit(jnp.asarray(result.x), jnp.asarray(y0), jnp.asarray(y1)))
            if new_value < old_value:
                best_result = result
        current = _reparameterize_coefficients(
            result.x, system, bath, k_modes, y0, y1, metric_batch
        )

    _, jnp = _import_jax()
    final_length = float(
        length_jit(jnp.asarray(best_result.x), jnp.asarray(y0), jnp.asarray(y1))
    )
    return final_length, np.asarray(best_result.x), best_result


def evaluate_fourier_length(
    system: SystemSpec,
    bath: Bath,
    lam: float,
    k_modes: int,
    quadrature_nodes: int,
    coefficients: np.ndarray,
) -> float:
    _, _, length_jit, _ = make_fourier_problem(system, bath, k_modes, quadrature_nodes)
    _, jnp = _import_jax()
    gaps = np.asarray(system.gaps)
    y0 = np.log(gaps)
    y1 = np.log(lam * gaps)
    return float(length_jit(jnp.asarray(coefficients), jnp.asarray(y0), jnp.asarray(y1)))



def make_polyline_problem(
    system: SystemSpec,
    bath: Bath,
    segments: int,
    quadrature_per_segment: int,
):
    """Independent piecewise-linear finite-element length functional."""
    jax, jnp = _import_jax()
    n = system.d - 1
    _, _, _, metric_batch = make_fourier_problem(system, bath, 1, 8)
    gl_x, gl_w = leggauss(quadrature_per_segment)
    t = jnp.asarray((gl_x + 1.0) / 2.0)
    weights = jnp.asarray(gl_w / 2.0)

    def segment_lengths(x, y0, y1):
        interior = x.reshape((segments - 1, n))
        nodes = jnp.concatenate([y0[None, :], interior, y1[None, :]], axis=0)
        displacement = nodes[1:] - nodes[:-1]
        samples = nodes[:-1, None, :] + t[None, :, None] * displacement[:, None, :]
        metric = metric_batch(samples.reshape((segments * quadrature_per_segment, n)))
        metric = metric.reshape((segments, quadrature_per_segment, n, n))
        speed2 = jnp.einsum("mi,mqij,mj->mq", displacement, metric, displacement)
        return jnp.sum(weights[None, :] * jnp.sqrt(jnp.maximum(speed2, 1e-300)), axis=1)

    def total_length(x, y0, y1):
        return jnp.sum(segment_lengths(x, y0, y1))

    return (
        jax.jit(jax.value_and_grad(total_length)),
        jax.jit(total_length),
        jax.jit(segment_lengths),
        metric_batch,
    )


def _polyline_initialization_from_fourier(
    coefficients: np.ndarray,
    system: SystemSpec,
    bath: Bath,
    lam: float,
    k_modes: int,
    segments: int,
    metric_batch,
) -> np.ndarray:
    _, jnp = _import_jax()
    n = system.d - 1
    gaps = np.asarray(system.gaps, dtype=float)
    y0 = np.log(gaps)
    y1 = np.log(lam * gaps)
    s = np.linspace(0.0, 1.0, 10000)
    y, ydot = _evaluate_fourier_path(coefficients, n, k_modes, y0, y1, s)
    metric = np.asarray(metric_batch(jnp.asarray(y)))
    speed = np.sqrt(np.maximum(np.einsum("bi,bij,bj->b", ydot, metric, ydot), 1e-300))
    arc = np.r_[0.0, cumulative_trapezoid(speed, s)]
    arc /= arc[-1]
    keep = np.r_[True, np.diff(arc) > 1e-12]
    u = np.linspace(0.0, 1.0, segments + 1)
    nodes = np.stack([np.interp(u, arc[keep], y[keep, a]) for a in range(n)], axis=1)
    return nodes[1:-1].ravel()


def _redistribute_polyline(
    x: np.ndarray,
    system: SystemSpec,
    lam: float,
    segments: int,
    metric_batch,
) -> np.ndarray:
    _, jnp = _import_jax()
    n = system.d - 1
    gaps = np.asarray(system.gaps, dtype=float)
    y0 = np.log(gaps)
    y1 = np.log(lam * gaps)
    nodes = np.vstack([y0, np.asarray(x).reshape((segments - 1, n)), y1])
    midpoint = 0.5 * (nodes[:-1] + nodes[1:])
    displacement = nodes[1:] - nodes[:-1]
    metric = np.asarray(metric_batch(jnp.asarray(midpoint)))
    approximate_length = np.sqrt(
        np.maximum(np.einsum("bi,bij,bj->b", displacement, metric, displacement), 1e-300)
    )
    arc = np.r_[0.0, np.cumsum(approximate_length)]
    arc /= arc[-1]
    u = np.linspace(0.0, 1.0, segments + 1)
    redistributed = np.stack([np.interp(u, arc, nodes[:, a]) for a in range(n)], axis=1)
    return redistributed[1:-1].ravel()


def optimize_polyline_path(
    system: SystemSpec,
    bath: Bath,
    lam: float,
    k_modes: int,
    fourier_coefficients: np.ndarray,
    segments: int,
    quadrature_per_segment: int,
    maxiter: int,
    cycles: int,
) -> tuple[float, np.ndarray, float]:
    value_grad, length_jit, segment_jit, metric_batch = make_polyline_problem(
        system, bath, segments, quadrature_per_segment
    )
    _, jnp = _import_jax()
    gaps = np.asarray(system.gaps, dtype=float)
    y0 = np.log(gaps)
    y1 = np.log(lam * gaps)
    y0j = jnp.asarray(y0)
    y1j = jnp.asarray(y1)
    x = _polyline_initialization_from_fourier(
        fourier_coefficients, system, bath, lam, k_modes, segments, metric_batch
    )

    def fun(v):
        value, grad = value_grad(jnp.asarray(v), y0j, y1j)
        return float(value), np.asarray(grad, dtype=float)

    best = None
    for cycle in range(cycles + 1):
        result = minimize(
            fun,
            x,
            jac=True,
            method="L-BFGS-B",
            options={
                "ftol": 1e-14,
                "gtol": 1e-8,
                "maxiter": maxiter,
                "maxls": 60,
                "maxcor": 50,
            },
        )
        value = float(length_jit(jnp.asarray(result.x), y0j, y1j))
        if best is None or value < best[0]:
            best = (value, np.asarray(result.x), result)
        if cycle < cycles:
            x = _redistribute_polyline(result.x, system, lam, segments, metric_batch)
    segment_values = np.asarray(segment_jit(jnp.asarray(best[1]), y0j, y1j))
    segment_cv = float(np.std(segment_values) / np.mean(segment_values))
    return best[0], best[1], segment_cv


def compute_branch(
    lambdas: np.ndarray,
    system: SystemSpec,
    bath: Bath,
    k_modes: int,
    quadrature_nodes: int,
    x_start: np.ndarray,
    direction: str,
    energy_iterations: int,
    length_iterations: int,
):
    order = range(len(lambdas)) if direction == "forward" else range(len(lambdas) - 1, -1, -1)
    n = system.d - 1
    x = np.asarray(x_start, dtype=float)
    lengths = np.full(len(lambdas), np.nan)
    coefficients: list[np.ndarray | None] = [None] * len(lambdas)

    for index in order:
        lam = float(lambdas[index])
        if abs(lam - 1.0) < 1e-14:
            x = np.zeros(n * k_modes)
            lengths[index] = 0.0
            coefficients[index] = x.copy()
            continue
        length, x, _ = optimize_fourier_path(
            system,
            bath,
            lam,
            k_modes,
            quadrature_nodes,
            x,
            energy_iterations=energy_iterations,
            length_iterations=length_iterations,
            reparameterization_cycles=1,
        )
        lengths[index] = length
        coefficients[index] = x.copy()
        print(f"{system.name:>2s} {direction:>8s} lambda={lam:5.2f}  L={length:.10f}")
    return lengths, np.stack(coefficients)


def default_lambda_grid(quick: bool = False) -> np.ndarray:
    if quick:
        return np.unique(np.r_[np.linspace(1.0, 3.0, 6), np.linspace(3.4, 5.0, 9), np.linspace(5.5, 9.0, 6)])
    return np.unique(
        np.r_[
            np.linspace(1.0, 3.0, 9),
            np.linspace(3.2, 5.2, 21),
            np.linspace(5.5, 9.0, 8),
        ]
    )


def save_curve_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    lambdas = data["lambdas"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lambda=T_initial/T_final",
                "T_final_over_T_initial",
                "L_d3_structured",
                "L_d4_structured",
                "L_d3_ohmic_control",
                "L_d4_ohmic_control",
            ]
        )
        for row in zip(
            lambdas,
            1.0 / lambdas,
            data["L3_structured"],
            data["L4_structured"],
            data["L3_ohmic"],
            data["L4_ohmic"],
        ):
            writer.writerow([f"{float(value):.16g}" for value in row])


def make_plots(curve_file: Path, output_dir: Path) -> None:
    from birth_death_plot import make_plot
    make_plot(curve_file, output_dir)


def verify(seed_file: Path, output_dir: Path, independent: bool, quick: bool) -> None:
    seeds = np.load(seed_file)
    cases = [
        (D3, LAMBDA_NEAR, int(seeds["d3_K"][0]), np.asarray(seeds["d3_near_coeff"]), "d3_near"),
        (D3, LAMBDA_FAR, int(seeds["d3_K"][0]), np.asarray(seeds["d3_far_coeff"]), "d3_far"),
        (D4, LAMBDA_NEAR, int(seeds["d4_K"][0]), np.asarray(seeds["d4_near_coeff"]), "d4_near"),
        (D4, LAMBDA_FAR, int(seeds["d4_K"][0]), np.asarray(seeds["d4_far_coeff"]), "d4_far"),
    ]

    result: dict[str, object] = {"quadrature": {}, "refined": {}}
    for system, lam, k_modes, coefficients, label in cases:
        table = {}
        for nodes in (128, 256, 512, 1024):
            table[str(nodes)] = evaluate_fourier_length(
                system, BATH, lam, k_modes, nodes, coefficients
            )
        result["quadrature"][label] = table
        print(f"{label}: " + "  ".join(f"N={n}: {v:.12f}" for n, v in table.items()))

        if quick:
            k_refine = min(k_modes, 18 if system.d == 4 else 20)
            energy_iterations, length_iterations, cycles = 1200, 700, 1
            quadrature_nodes = 384
        else:
            k_refine = k_modes
            energy_iterations, length_iterations, cycles = 3500, 2200, 2
            quadrature_nodes = 640 if system.d == 4 else 768
        x0 = _resize_coefficients(coefficients, system.d - 1, k_modes, k_refine)
        length, refined, opt = optimize_fourier_path(
            system,
            BATH,
            lam,
            k_refine,
            quadrature_nodes,
            x0,
            energy_iterations,
            length_iterations,
            reparameterization_cycles=cycles,
        )
        result["refined"][label] = {
            "K": k_refine,
            "quadrature_nodes": quadrature_nodes,
            "length": length,
            "optimizer_success": bool(opt.success),
            "gradient_norm": float(np.linalg.norm(opt.jac)),
        }
        print(f"{label}: refined L={length:.12f}, |grad|={np.linalg.norm(opt.jac):.3e}")

    d3_near = result["refined"]["d3_near"]["length"]
    d3_far = result["refined"]["d3_far"]["length"]
    d4_near = result["refined"]["d4_near"]["length"]
    d4_far = result["refined"]["d4_far"]["length"]
    result["inversion"] = {
        "d3": {"near": d3_near, "far": d3_far, "gap": d3_near - d3_far},
        "d4": {"near": d4_near, "far": d4_far, "gap": d4_near - d4_far},
    }
    print(json.dumps(result["inversion"], indent=2))

    if independent:
        result["polyline"] = {}
        for system, lam, k_modes, coefficients, label in cases:
            if quick:
                segments = 45 if system.d == 3 else 40
                q_segment = 8
                maxiter = 1400
                cycles = 1
            else:
                segments = 90 if system.d == 3 else 70
                q_segment = 12 if system.d == 3 else 10
                maxiter = 3200 if system.d == 3 else 3000
                cycles = 2
            value, _, segment_cv = optimize_polyline_path(
                system,
                BATH,
                lam,
                k_modes,
                coefficients,
                segments,
                q_segment,
                maxiter,
                cycles,
            )
            result["polyline"][label] = {
                "segments": segments,
                "quadrature_per_segment": q_segment,
                "length": value,
                "segment_length_cv": segment_cv,
            }
            print(
                f"{label}: polyline L={value:.12f}, segments={segments}, "
                f"segment-CV={segment_cv:.3e}"
            )
        poly_d3_gap = (
            result["polyline"]["d3_near"]["length"]
            - result["polyline"]["d3_far"]["length"]
        )
        poly_d4_gap = (
            result["polyline"]["d4_near"]["length"]
            - result["polyline"]["d4_far"]["length"]
        )
        result["polyline_inversion"] = {"d3_gap": poly_d3_gap, "d4_gap": poly_d4_gap}
        print(json.dumps(result["polyline_inversion"], indent=2))

    output_path = output_dir / "verification_rerun.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output_path}")


def recompute(seed_file: Path, output_dir: Path, quick: bool) -> Path:
    seeds = np.load(seed_file)
    lambdas = default_lambda_grid(quick=quick)

    if quick:
        settings = {
            "d3": (12, 224, 900, 350),
            "d4": (12, 224, 1000, 350),
            "ref": (10, 192, 800, 300),
        }
    else:
        settings = {
            "d3": (18, 384, 1800, 600),
            "d4": (16, 320, 1500, 450),
            "ref": (12, 256, 1200, 350),
        }

    k3, q3, e3, l3 = settings["d3"]
    k4, q4, e4, l4 = settings["d4"]
    kr, qr, er, lr = settings["ref"]

    L3_forward, C3_forward = compute_branch(
        lambdas, D3, BATH, k3, q3, np.zeros((D3.d - 1) * k3), "forward", e3, l3
    )
    d3_far_seed = _resize_coefficients(
        np.asarray(seeds["d3_far_coeff"]), D3.d - 1, int(seeds["d3_K"][0]), k3
    )
    L3_backward, C3_backward = compute_branch(
        lambdas, D3, BATH, k3, q3, d3_far_seed, "backward", e3, l3
    )

    L4_forward, C4_forward = compute_branch(
        lambdas, D4, BATH, k4, q4, np.zeros((D4.d - 1) * k4), "forward", e4, l4
    )
    d4_far_seed = _resize_coefficients(
        np.asarray(seeds["d4_far_coeff"]), D4.d - 1, int(seeds["d4_K"][0]), k4
    )
    L4_backward, C4_backward = compute_branch(
        lambdas, D4, BATH, k4, q4, d4_far_seed, "backward", e4, l4
    )

    L3_ref, C3_ref = compute_branch(
        lambdas, D3, OHMIC_CONTROL, kr, qr, np.zeros((D3.d - 1) * kr), "forward", er, lr
    )
    L4_ref, C4_ref = compute_branch(
        lambdas, D4, OHMIC_CONTROL, kr, qr, np.zeros((D4.d - 1) * kr), "forward", er, lr
    )

    curve_path = output_dir / "thermodynamic_mpemba_curve_paths_recomputed.npz"
    np.savez_compressed(
        curve_path,
        lambdas=lambdas,
        temperatures=1.0 / lambdas,
        L3_structured=np.minimum(L3_forward, L3_backward),
        L3_structured_forward=L3_forward,
        L3_structured_backward=L3_backward,
        C3_structured_forward=C3_forward,
        C3_structured_backward=C3_backward,
        L4_structured=np.minimum(L4_forward, L4_backward),
        L4_structured_forward=L4_forward,
        L4_structured_backward=L4_backward,
        C4_structured_forward=C4_forward,
        C4_structured_backward=C4_backward,
        L3_ohmic=L3_ref,
        C3_ohmic=C3_ref,
        L4_ohmic=L4_ref,
        C4_ohmic=C4_ref,
        d3_gaps=np.asarray(D3.gaps),
        d3_couplings=np.asarray(D3.couplings),
        d4_gaps=np.asarray(D4.gaps),
        d4_couplings=np.asarray(D4.couplings),
        bath_params=np.asarray([BATH.A, BATH.gamma, BATH.Omega, BATH.omega_c]),
    )
    print(f"Wrote {curve_path}")
    return curve_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plot", "verify", "recompute"), default="plot")
    parser.add_argument("--quick", action="store_true", help="Use a reduced basis/iteration count.")
    parser.add_argument(
        "--independent",
        action="store_true",
        help="Report the independent finite-element certification stored in the JSON file.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed-file", type=Path, default=None)
    parser.add_argument("--curve-file", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve() if args.output_dir else script_dir.parent / "figures" / "birth-death"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_file = args.seed_file or (script_dir / "data" / "birth-death" / "thermodynamic_mpemba_seed_paths.npz")
    curve_file = args.curve_file or (script_dir / "data" / "birth-death" / "thermodynamic_length_curves.csv")

    if args.mode in ("verify", "recompute") and not seed_file.exists():
        raise FileNotFoundError("Original path coefficients are absent. Supply --seed-file to rerun the original solver; plot mode uses the preserved CSV without changing simulations.")

    if args.mode == "recompute":
        curve_file = recompute(seed_file, output_dir, quick=args.quick)
    elif args.mode == "verify":
        verify(seed_file, output_dir, independent=args.independent, quick=args.quick)

    # All modes end by producing/refreshing plots.
    make_plots(curve_file, output_dir)
    print(f"Figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
