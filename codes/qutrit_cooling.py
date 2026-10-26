#!/usr/bin/env python3
"""Reversible constant-eigenrate qutrit: one cooling model, two figures.

Run from any directory. Outputs are relative to this repository's figures/qutrit-cooling directory.
Python >= 3.9; NumPy, SciPy and Matplotlib. No external data are required.
The boundary-value solver is initialized with a straight line in angle space,
independently of the exact great-circle construction.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class Model:
    energies: tuple = (0.0, 0.03, 1.0)
    beta: float = 0.1             # Fixed bath inverse temperature; Ti = 1/beta.
    r_theta: float = 5.0         # Physical relaxation eigenrates, not metric entries.
    r_phi: float = 100.0
    near_scale: float = 53.62484152434405      # H_final = scale * H_reference.
    far_scale: float = 206.0341352609942
    max_scale: float = 300.0

    def __post_init__(self):
        if not (self.beta > 0 and 0 < self.r_theta <= self.r_phi):
            raise ValueError("Require beta > 0 and 0 < r_theta <= r_phi.")
        if not (1 < self.near_scale < self.far_scale <= self.max_scale):
            raise ValueError("Require cooling endpoints 1 < near < far <= max.")

    @property
    def kappa(self):
        return np.sqrt(self.r_theta / self.r_phi)

    @property
    def radius(self):
        return 2.0 / np.sqrt(self.beta * self.r_theta)


def gibbs(scale, model):
    """Equilibrium population vector of scale * H0 at the fixed bath beta."""
    logits = -model.beta * float(scale) * np.asarray(model.energies)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def angles(q):
    """Pole is level 1: q=(sin(theta)cos(phi), cos(theta), sin(theta)sin(phi))."""
    q = np.asarray(q)
    return np.array([np.arctan2(np.hypot(q[0], q[2]), q[1]),
                     np.arctan2(q[2], q[0])])


def sphere(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.cos(theta),
                     np.sin(theta) * np.sin(phi)])


def frame(theta, phi):
    a = np.array([np.cos(theta) * np.cos(phi), -np.sin(theta),
                  np.cos(theta) * np.sin(phi)])
    b = np.array([-np.sin(phi), 0.0, np.cos(phi)])
    return a, b


def generator(q, model):
    """Column convention: z'=Mz in physical time; M_ij is j -> i."""
    a, b = frame(*angles(q))
    R = model.r_theta * np.outer(a, a) + model.r_phi * np.outer(b, b)
    M = -q[:, None] * R / q[None, :]
    return M, R


def auxiliary(theta, phi, model):
    psi = model.kappa * phi
    return np.array([np.cos(theta), np.sin(theta) * np.cos(psi),
                     np.sin(theta) * np.sin(psi)])


def central_angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))


def exact_distance(scale, model):
    a = auxiliary(*angles(np.sqrt(gibbs(1, model))), model)
    b = auxiliary(*angles(np.sqrt(gibbs(scale, model))), model)
    return model.radius * central_angle(a, b)


def exact_path(scale, model, u):
    """Constant thermodynamic speed; the convex auxiliary wedge fixes the branch."""
    a = auxiliary(*angles(np.sqrt(gibbs(1, model))), model)
    b = auxiliary(*angles(np.sqrt(gibbs(scale, model))), model)
    alpha = central_angle(a, b)
    u = np.asarray(u)
    if alpha < 1e-14:
        v = np.repeat(a[:, None], u.size, axis=1)
    else:
        v = (a[:, None] * np.sin((1-u)*alpha)
             + b[:, None] * np.sin(u*alpha)) / np.sin(alpha)
    theta = np.arctan2(np.hypot(v[1], v[2]), v[0])
    phi = np.arctan2(v[2], v[1]) / model.kappa
    if np.any(phi < -1e-12) or np.any(phi > np.pi/2 + 1e-12):
        raise RuntimeError("The auxiliary arc left the physical chart.")
    return sphere(theta, phi), np.array([theta, phi])


def numerical_path(scale, model, tolerance=1e-9):
    """Solve the four first-order geodesic ODEs with fixed angular endpoints."""
    start = angles(np.sqrt(gibbs(1, model)))
    end = angles(np.sqrt(gibbs(scale, model)))
    u = np.linspace(0, 1, 61)
    delta = end - start
    guess = np.vstack([start[:, None] + delta[:, None]*u,
                       np.repeat(delta[:, None], len(u), axis=1)])

    def rhs(_, y):
        theta, _, theta_prime, phi_prime = y
        return np.array([theta_prime, phi_prime,
                         model.kappa**2*np.sin(theta)*np.cos(theta)*phi_prime**2,
                         -2*np.cos(theta)/np.sin(theta)*theta_prime*phi_prime])

    def boundary(ya, yb):
        return np.concatenate([ya[:2] - start, yb[:2] - end])

    sol = solve_bvp(rhs, boundary, u, guess, tol=tolerance, max_nodes=20000)
    if not sol.success:
        raise RuntimeError(f"BVP failed at scale={scale}: {sol.message}")

    def speed(t):
        theta, _, dt, dp = sol.sol(t)
        return model.radius * np.sqrt(dt**2 + model.kappa**2*np.sin(theta)**2*dp**2)

    length, _ = quad(speed, 0, 1, epsabs=1e-11, epsrel=1e-11)
    return sol, length


def validate(model, scales):
    """Independent BVP, Markov constraints, spectral and metric identities."""
    rows, errors, path_errors = [], [], []
    residuals = dict(column_sum=0.0, stationary=0.0, detailed_balance=0.0,
                     drazin_identity=0.0, metric_identity=0.0, endpoint=0.0,
                     spectrum=0.0)
    minimum_rate = float("inf")
    rng = np.random.default_rng(2741)
    offdiag = ~np.eye(3, dtype=bool)
    for scale in scales:
        sol, length = numerical_path(float(scale), model)
        exact = exact_distance(scale, model)
        errors.append(abs(length - exact))
        u = np.linspace(0, 1, 151)
        q, _ = exact_path(scale, model, u)
        y = sol.sol(u)
        path_errors.append(float(np.max(np.abs(q-sphere(y[0], y[1])))))
        residuals['endpoint'] = max(residuals['endpoint'],
            float(np.max(np.abs(q[:, 0]**2-gibbs(1, model)))),
            float(np.max(np.abs(q[:, -1]**2-gibbs(scale, model)))))
        rows.append(dict(xi=float(1/scale),
                         exact_length=float(exact), bvp_length=float(length)))
        for qi in q[:, ::15].T:
            M, R = generator(qi, model)
            p = qi**2
            minimum_rate = min(minimum_rate, float(M[offdiag].min()))
            residuals['column_sum'] = max(residuals['column_sum'], float(abs(M.sum(axis=0)).max()))
            residuals['stationary'] = max(residuals['stationary'], float(abs(M@p).max()))
            flow = M*p[None, :]
            residuals['detailed_balance'] = max(residuals['detailed_balance'], float(abs(flow-flow.T).max()))
            residuals['spectrum'] = max(residuals['spectrum'], float(abs(np.linalg.eigvalsh(R)-[0,model.r_theta,model.r_phi]).max()))
            # Independently invert M on the zero-sum subspace, not via R's frame.
            projector = np.outer(p, np.ones(3))
            Mplus = np.linalg.inv(M+projector)-projector
            residuals['drazin_identity'] = max(residuals['drazin_identity'], float(abs(M@Mplus-(np.eye(3)-projector)).max()))
            v = rng.normal(size=3)
            v -= qi*np.dot(qi, v)
            dp = 2*qi*v
            direct = -np.dot(dp/p, Mplus@dp)/model.beta
            a, b = frame(*angles(qi))
            geometric = 4/model.beta*((a@v)**2/model.r_theta+(b@v)**2/model.r_phi)
            residuals['metric_identity'] = max(residuals['metric_identity'], abs(direct-geometric)/max(1, abs(geometric)))
    if max(errors) > 1e-7 or max(path_errors) > 1e-6:
        raise AssertionError("Analytic and numerical solutions disagree.")
    if minimum_rate < -1e-10 or max(residuals.values()) > 1e-8:
        raise AssertionError(f"Physical or metric checks failed: {residuals}")
    near, far = [exact_distance(r, model) for r in (model.near_scale, model.far_scale)]
    if not far < near:
        raise AssertionError("The selected cooling endpoints do not show inversion.")
    peak = minimize_scalar(lambda r: -exact_distance(r, model), bounds=(1, model.far_scale), method='bounded')
    trough = minimize_scalar(lambda r: exact_distance(r, model),
                             bounds=(peak.x, model.max_scale), method='bounded')
    if not (np.isclose(peak.x, model.near_scale, rtol=1e-5)
            and np.isclose(trough.x, model.far_scale, rtol=1e-5)):
        raise AssertionError("Selected endpoints must be the local distance extrema.")
    report = dict(model=asdict(model), max_bvp_length_error=max(errors),
                  max_bvp_path_error=max(path_errors), minimum_sampled_transition_rate=minimum_rate,
                  residuals=residuals, near_length=near, far_length=far,
                  dissipation_reduction=1-(far/near)**2,
                  local_peak_scale=float(peak.x), local_peak_length=float(-peak.fun),
                  xi_B=float(1/peak.x), xi_C=float(1/trough.x),
                  local_minimum_length=float(trough.fun),
                  bvp_count=len(rows), scope="Equilibrium-path geometry, not finite-time master-equation integration.")
    return rows, report


def make_figures(model, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class TangentArrow3D(FancyArrowPatch):
        """Project physical 3D endpoints, with equal-sized heads in points."""
        def __init__(self, start, end, **kwargs):
            super().__init__((0, 0), (0, 0), **kwargs)
            self.endpoints = np.column_stack((start, end))

        def do_3d_projection(self, renderer=None):
            x, y, z = proj3d.proj_transform(*self.endpoints, self.axes.get_proj())
            self.set_positions((x[0], y[0]), (x[1], y[1]))
            return float(np.min(z))

        def draw(self, renderer):
            self.do_3d_projection()
            super().draw(renderer)

    navy, teal, amber, red = '#19344A', '#007E87', '#CF8B24', '#C54E53'
    blue = '#2675B5'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16,
                         'axes.labelcolor': navy, 'text.color': navy,
                         'axes.edgecolor': '#A5B1B9', 'xtick.color': navy,
                         'ytick.color': navy, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})

    # Only reorder Cartesian axes for the drawing: x=q0, y=q2, z=q1.
    def xyz(q):
        return q[0], q[2], q[1]

    fig = plt.figure(figsize=(4.5, 3.8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    th, ph = np.meshgrid(np.linspace(0, np.pi/2, 50), np.linspace(0, np.pi/2, 50))
    ax.plot_surface(*xyz(sphere(th, ph)), color='#E8EEF1', alpha=.17,
                    linewidth=0, shade=False, zorder=0)
    for t in np.linspace(0, np.pi/2, 6)[1:]:
        ax.plot(*xyz(sphere(t*np.ones(160), np.linspace(0, np.pi/2, 160))),
                color='#BECBD2', lw=.6, alpha=.8, zorder=1)
    for f in np.linspace(0, np.pi/2, 6):
        ax.plot(*xyz(sphere(np.linspace(0, np.pi/2, 160), f*np.ones(160))),
                color='#BECBD2', lw=.6, alpha=.8, zorder=1)
    qred = np.array([np.sqrt(gibbs(r, model)) for r in np.r_[0, np.geomspace(.01, 3000, 1100)]]).T
    ax.plot(*xyz(qred), color=red, lw=2.0, zorder=5)
    for scale, color, label in [(model.near_scale, amber, r'$B$'),
                                 (model.far_scale, blue, r'$C$')]:
        sol, _ = numerical_path(scale, model)
        y = sol.sol(np.linspace(0, 1, 401))
        path = sphere(y[0], y[1])
        ax.plot(*xyz(path), color=color, lw=2.8, zorder=6)
        point = path[:, -1]
        ax.scatter(*xyz(point), color=color, s=42, depthshade=False, zorder=8)
        label_offset = (-.025, .018, .065) if label == r'$B$' else (-.060, .018, -.080)
        if label == r'$B$':
            ax.text(point[0]+label_offset[0], point[2]+label_offset[1], point[1]+label_offset[2], label,
                    color=color, fontsize=16, zorder=9)
        else:
            c_label_point = point.copy()
    # Tangents in the physical square-root probability embedding, at C.
    # The arriving geodesic is parametrized at constant thermodynamic speed.
    sol, distance = numerical_path(model.far_scale, model)
    theta, phi, dt, dp = sol.sol(1)
    point = sphere(theta, phi)
    a, b = frame(theta, phi)
    arriving = (a * dt + b * np.sin(theta) * dp) / distance
    energies = np.asarray(model.energies)
    cooling = -1 / 2 * (energies - energies @ (point**2)) * point
    def metric(u, v):
        return 4 / model.beta * ((a @ u)*(a @ v)/model.r_theta
                                + (b @ u)*(b @ v)/model.r_phi)
    angle = np.degrees(np.arccos(np.clip(
        metric(arriving, cooling) / np.sqrt(metric(arriving, arriving)
                                            * metric(cooling, cooling)), -1, 1)))
    for vector, color, label, length in [
            (arriving, blue, r'$\mathbf{v}_\varsigma$', .34),
            (cooling, red, r"$\mathbf{v}_\omega$", .34)]:
        arrow = length * vector / np.linalg.norm(vector)
        ax.add_artist(TangentArrow3D(
            xyz(point), xyz(point + arrow), color=color, linewidth=1.6,
            arrowstyle='-|>,head_length=0.55,head_width=0.28',
            mutation_scale=12, shrinkA=0, shrinkB=0, zorder=12))
        label_point = point + 1.15*arrow
        if color == red:
            label_point += np.array([.06, .015, 0.])
        else:
            label_point += np.array([0., .025, 0.])
        ax.text(*xyz(label_point), label, color=color,
                fontsize=16, ha='right' if color == red else 'center', zorder=13)
    (output/'sphere-endpoint-vectors.json').write_text(json.dumps(dict(
        scale=model.far_scale, point=point.tolist(),
        arriving_unit_tangent=arriving.tolist(), cooling_tangent=cooling.tolist(),
        metric_pairing=float(metric(arriving, cooling)),
        metric_angle_degrees=float(angle)), indent=2)+'\n')
    initial = np.sqrt(gibbs(1, model))
    ax.scatter(*xyz(initial), color=navy, s=48, depthshade=False, zorder=9)
    ax.text(initial[0]-.025, initial[2]+.018, initial[1]+.035, '$A$', color=navy, fontsize=16)
    for vec, label in [(np.array([1.08,0,0]), r'$\sqrt{\omega_1}$'),
                        (np.array([0,1.08,0]), r'$\sqrt{\omega_3}$'),
                        (np.array([0,0,1.08]), r'$\sqrt{\omega_2}$')]:
        ax.quiver(0,0,0,*vec, color=navy, linewidth=.8, arrow_length_ratio=.04)
        label_position = vec.copy()
        if vec[2] == 0:
            label_position[2] -= .135
        else:
            label_position[2] += .055
        ax.text(*label_position, label, fontsize=16, ha='center')
    # Coordinate guide in the free region near the origin.  The polar
    # axis is sqrt(omega_2), and azimuth starts at sqrt(omega_1).
    guide_color = '#675180'
    guide_theta, guide_phi, guide_radius = 1.05, 1.50, 0.43
    tip = guide_radius * sphere(guide_theta, guide_phi)
    projection = tip.copy()
    projection[1] = 0
    ax.add_artist(TangentArrow3D(
        (0, 0, 0), xyz(tip), color=guide_color, linewidth=1.3,
        arrowstyle='-|>', mutation_scale=11, shrinkA=0, shrinkB=0, zorder=15))
    ax.plot(*xyz(np.column_stack([np.zeros(3), projection, tip])),
            color=guide_color, lw=1, ls=(0, (3, 2)), zorder=14)
    polar_arc = 0.27 * sphere(np.linspace(0, guide_theta, 50), guide_phi)
    azimuth_arc = 0.23 * sphere(np.full(50, np.pi/2), np.linspace(0, guide_phi, 50))
    ax.plot(*xyz(polar_arc), color=guide_color, lw=1, zorder=15)
    ax.plot(*xyz(azimuth_arc), color=guide_color, lw=1, zorder=15)
    ax.text(*xyz(0.34 * sphere(guide_theta/2, guide_phi)),
            r'$\theta$', color=guide_color, fontsize=13, zorder=16)
    ax.text(*xyz(0.30 * sphere(np.pi/2, guide_phi/2)),
            r'$\varphi$', color=guide_color, fontsize=13, va='top', zorder=16)
    ax.view_init(elev=20, azim=65)
    ax.set(xlim=(0,1.12), ylim=(0,1.12), zlim=(0,1.12))
    ax.set_box_aspect((1,1,1), zoom=1.02)
    cx, cy, _ = proj3d.proj_transform(*xyz(c_label_point), ax.get_proj())
    ax.annotate('$C$', (cx, cy), xytext=(-12, -2), textcoords='offset points',
                ha='right', va='center', color=blue, fontsize=16, zorder=14)
    ax.set_axis_off()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    save_figure(fig, output/'qutrit-cooling-sphere')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    temp = np.geomspace(1,model.max_scale,600)
    lengths = np.array([exact_distance(t,model) for t in temp])
    ax.plot(temp, lengths, color=navy, lw=2, label='Exact geodesic distance')
    tn, tf = model.near_scale, model.far_scale
    ax.axvspan(tn,tf,color=teal,alpha=.065,lw=0)
    for t, scale, color, label, offset in [(tn,model.near_scale,amber,'B',(8,8)),
                                          (tf,model.far_scale,blue,'C',(8,-16))]:
        d = exact_distance(scale,model)
        ax.vlines(t,0,d,color=color,lw=1,linestyle=(0,(3,3)),alpha=.8)
        ax.scatter(t,d,s=55,color=color,zorder=5)
        ax.annotate(label,(t,d),xytext=offset,textcoords='offset points',color=color,fontsize=16)
    ax.scatter(1,0,s=40,color=navy,zorder=5,clip_on=False)
    ax.annotate('A',(1,0),xytext=(8,10),textcoords='offset points',fontsize=16)
    ax.set(xscale='log', xlim=(.95,model.max_scale*1.05),ylim=(0,lengths.max()*1.12),
           xlabel=r'$\beta^\prime/\beta$', ylabel=r'$L(\beta\rightarrow\beta^\prime)$')
    ax.annotate('cooling',xy=(.51,.93),xytext=(.18,.93),xycoords='axes fraction',
                textcoords='axes fraction',ha='center',va='center',color='#687D89',
                arrowprops=dict(arrowstyle='->',color='#687D89',lw=1))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='y',color='#DCE4E8',lw=.65)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.23,right=.96,bottom=.22,top=.94)
    save_figure(fig,output/'qutrit-cooling-distance')
    plt.close(fig)


def save_figure(fig, base):
    # Crop to actual rendered marks, including 3D labels, rather than the
    # axes rectangle (which retains large empty margins for the octant).
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    yy, xx = np.nonzero(np.any(rgba[:, :, :3] < 250, axis=2))
    height = rgba.shape[0]
    pad = 2 / 72  # Two typographic points beyond the visible content.
    bounds = Bbox.from_extents(xx.min()/fig.dpi-pad,
                              (height-yy.max()-1)/fig.dpi-pad,
                              (xx.max()+1)/fig.dpi+pad,
                              (height-yy.min())/fig.dpi+pad)
    for suffix in ('.pdf','.svg','.png'):
        fig.savefig(base.with_suffix(suffix),dpi=220,bbox_inches=bounds,
                    facecolor='white')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'figures'/'qutrit-cooling')
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    model = Model()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    scales = np.unique(np.r_[1/np.geomspace(1/model.max_scale,1,31),model.near_scale,model.far_scale])
    rows, report = validate(model,scales)
    (args.output_dir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output_dir/'distance-data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir/'geodesic-paths.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target', 'xi', 's', 'theta', 'phi',
                         'omega0', 'omega1', 'omega2', 'epsilon1', 'epsilon2'])
        for label, scale in [('B', model.near_scale), ('C', model.far_scale)]:
            sol, _ = numerical_path(scale, model)
            for s in np.linspace(0, 1, 401):
                theta, phi = sol.sol(s)[:2]
                p = sphere(theta, phi)**2
                energies = -np.log(p[1:]/p[0])/model.beta
                writer.writerow([label, 1/scale, s, theta, phi, *p, *energies])
    if not args.verify_only:
        make_figures(model,rows,args.output_dir)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
