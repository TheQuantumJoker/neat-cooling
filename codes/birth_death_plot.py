"""Birth--death panel for the combined three-panel figure; preserve the supplied numerical samples."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

def make_plot(curve_file, output_dir, certification_file=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, NullFormatter, MaxNLocator
    from qutrit_cooling import save_figure

    curve_file, output_dir = Path(curve_file), Path(output_dir)
    if curve_file.suffix == '.npz':
        with np.load(curve_file) as raw:
            xi = 1 / raw['lambdas']
            values = {d: [raw[f'L{d}_structured'].copy(), raw[f'L{d}_ohmic'].copy()]
                      for d in (3, 4)}
    else:
        with curve_file.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        xi = np.array([float(r['T_final_over_T_initial']) for r in rows])
        values = {d: [np.array([float(r[f'L_d{d}_{bath}']) for r in rows])
                      for bath in ('structured', 'ohmic_control')] for d in (3, 4)}
        scales = np.array([float(r['lambda=T_initial/T_final']) for r in rows])
        if not np.allclose(xi * scales, 1, rtol=1e-14):
            raise ValueError('Inconsistent endpoint temperature coordinates.')
    order = np.argsort(xi)
    xi = xi[order]
    if np.any(np.diff(xi) <= 0) or np.any(xi <= 0):
        raise ValueError('Expected distinct positive temperatures.')
    values = {d: [v[order] for v in curves] for d, curves in values.items()}
    if any(not np.all(np.isfinite(v) & (v >= 0)) for curves in values.values() for v in curves):
        raise ValueError('Lengths must be finite and nonnegative.')

    navy, teal, amber = '#19344A', '#007E87', '#CF8B24'
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':16,
        'axes.labelcolor':navy, 'text.color':navy, 'axes.edgecolor':'#A5B1B9',
        'xtick.color':navy, 'ytick.color':navy, 'pdf.fonttype':42, 'ps.fonttype':42,
        'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    for d, color in [(3, teal), (4, amber)]:
        ax.plot(1/xi, values[d][0], color=color, lw=2)
        ax.plot(1/xi, values[d][1], color=color, lw=1.7, linestyle=(0,(4,2.5)))

    certification_file = Path(certification_file) if certification_file else curve_file.with_name('thermodynamic_length_certification.json')
    report = {'source':curve_file.name,
              'source_sha256':hashlib.sha256(curve_file.read_bytes()).hexdigest(),
              'samples_per_curve':len(xi), 'interpolation':'none; straight segments between original samples',
              'x_definition':"beta_prime / beta = T_i / T_f",
              'checks':{}}
    near, far = 4.25, 9
    ax.axvspan(near,far,color=teal,alpha=.065,lw=0)
    for x in (far,near):
        ax.axvline(x,color='#A5B1B9',lw=.7,linestyle=(0,(2,3)),zorder=0)
    if certification_file.exists():
        certified = json.loads(certification_file.read_text())['certified_pair']
        for d, color in [(3,teal),(4,amber)]:
            near_value, far_value = [certified[f'd{d}'][k] for k in ('L_near','L_far')]
            if not far_value < near_value:
                raise ValueError('Certified endpoints do not show inversion.')
            # As in the original figure, highlights use the separately certified pair.
            ax.scatter([near,far],[near_value,far_value],color=color,s=25,zorder=4)
            report['checks'][f'd{d}_inversion_gap'] = near_value-far_value
    for d in (3,4):
        if np.any(np.diff(values[d][1]) > 1e-10):
            raise ValueError('Ohmic control no longer monotone along cooling.')
        report['checks'][f'd{d}_ohmic_monotone'] = True
    ymax = max(v.max() for curves in values.values() for v in curves)
    ax.set(xscale='log',xlim=(.95,10),ylim=(0,1.17*ymax),
           xlabel=r'$\beta^\prime/\beta$',ylabel=r'$L(\beta\rightarrow\beta^\prime)$')
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.annotate('cooling',xy=(.51,.93),xytext=(.18,.93),xycoords='axes fraction',
                ha='center',va='center',color='#687D89',
                arrowprops=dict(arrowstyle='->',color='#687D89',lw=1))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='y',color='#DCE4E8',lw=.65)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.23,right=.96,bottom=.22,top=.94)
    output_dir.mkdir(parents=True,exist_ok=True)
    save_figure(fig,output_dir/'birth-death-cooling')
    plt.close(fig)
    (output_dir/'plot-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__ == '__main__':
    import argparse
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curve-file',type=Path,default=root/'codes/data/birth-death/thermodynamic_length_curves.csv')
    parser.add_argument('--output-dir',type=Path,default=root/'figures/birth-death')
    args=parser.parse_args()
    print(json.dumps(make_plot(args.curve_file,args.output_dir),indent=2))
