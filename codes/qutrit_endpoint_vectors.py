"""Illustrate the two endpoint tangents using the existing cooling model."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qutrit_cooling import Model, gibbs, angles, sphere, frame, numerical_path, exact_distance

model = Model()
xi = 0.01
r = 1 / xi
sol, D = numerical_path(r, model)
theta, phi, dt, df = sol.sol(1)
q = sphere(theta, phi)
a, b = frame(theta, phi)
velocity = a * dt + b * np.sin(theta) * df
T = velocity / D
energies = np.asarray(model.energies)
cool = -model.beta / (2 * xi**2) * (energies - energies @ (q*q)) * q

def metric_components(v):
    return 2 / np.sqrt(model.beta) * np.array([
        a @ v / np.sqrt(model.r_theta), b @ v / np.sqrt(model.r_phi)])

t = metric_components(T)
c = metric_components(cool)
dot = t @ c
angle = np.degrees(np.arccos(np.clip(dot / np.linalg.norm(c), -1, 1)))
h = 1e-7
derivative = (exact_distance(1/(xi+h), model)-exact_distance(1/(xi-h), model))/(2*h)
assert np.isclose(t @ t, 1, atol=1e-8)
assert np.isclose(-dot, derivative, rtol=1e-6)

navy, teal, red = '#19344A', '#007E87', '#C54E53'
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans',
                     'text.color': navy, 'axes.labelcolor': navy, 'pdf.fonttype': 42})
fig = plt.figure(figsize=(11, 4.8))
ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
xyz = lambda v: (v[0], v[2], v[1])
u = np.linspace(0, np.pi/2, 140)
for th in np.linspace(0, np.pi/2, 7)[1:]:
    ax.plot(*xyz(sphere(th+0*u, u)), color='#CBD7DD', lw=.6)
for ph in np.linspace(0, np.pi/2, 7):
    ax.plot(*xyz(sphere(u, ph+0*u)), color='#CBD7DD', lw=.6)
family = np.array([np.sqrt(gibbs(x, model)) for x in np.geomspace(.05, 2500, 700)]).T
ax.plot(*xyz(family), color=red, lw=2)
path = sphere(*sol.sol(np.linspace(0, 1, 401))[:2])
ax.plot(*xyz(path), color=teal, lw=2.5)
start = path[:, 0]
ax.scatter(*xyz(start), s=30, color=navy, depthshade=False)
ax.text(*xyz(start+np.array([-.015,.045,.015])), '$A$', fontsize=12)
ax.scatter(*xyz(q), s=35, color=navy, depthshade=False, zorder=10)
ax.text(*xyz(q+np.array([-.025,.045,.025])), r'$P=\gamma(\xi)$', fontsize=11)
for v, color, length in [(T,teal,.23),(cool,red,.23)]:
    v = v / np.linalg.norm(v) * length
    ax.quiver(*xyz(q), *xyz(v), color=color, linewidth=2, arrow_length_ratio=.22, zorder=15)
ax.text(*xyz(q+T/np.linalg.norm(T)*.27), '$T$', color=teal, fontsize=13)
ax.text(*xyz(q+cool/np.linalg.norm(cool)*.29), r'$v_{\rm cool}$', color=red, fontsize=12)
for v,label in [(np.array([1.08,0,0]),'$q_0$'),(np.array([0,1.08,0]),'$q_1$'),(np.array([0,0,1.08]),'$q_2$')]:
    ax.quiver(0,0,0,*xyz(v),color=navy,lw=.7,arrow_length_ratio=.035)
    ax.text(*xyz(v),label,fontsize=11)
ax.view_init(elev=20, azim=65)
ax.set(xlim=(0,1.12),ylim=(0,1.12),zlim=(0,1.12))
ax.set_box_aspect((1,1,1),zoom=1.12)
ax.set_axis_off()
ax.set_title('Probability octant',loc='left',pad=15)

ax2 = fig.add_subplot(122)
# Rotate metric-orthonormal components so T points right. This preserves angles.
t = t / np.linalg.norm(t)
c = c / np.linalg.norm(c)
c = np.array([t @ c, np.linalg.det(np.vstack([t,c]))])
if c[1] < 0:
    c[1] *= -1
ax2.axhline(0,color='#D7E0E5',lw=.8)
ax2.axvline(0,color='#D7E0E5',lw=.8)
for v,color,label in [(np.array([1.,0]),teal,'$T$'),(c,red,r'$v_{\rm cool}/\|v_{\rm cool}\|_m$')]:
    ax2.annotate('',xy=v,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=color,lw=2.5,mutation_scale=16))
    ax2.text(v[0]*1.09,v[1]*1.1+.04,label,color=color,ha='center',fontsize=12)
arc = np.linspace(0,np.arccos(c[0]),100)
ax2.plot(.38*np.cos(arc),.38*np.sin(arc),color=navy,lw=1)
ax2.text(.32,.39,rf'$\alpha={angle:.1f}^\circ$',fontsize=12)
ax2.scatter(0,0,color=navy,s=28,zorder=4)
ax2.text(-.08,-.13,'$P$',fontsize=12)
ax2.set(xlim=(-1.25,1.25),ylim=(-.55,1.3),aspect='equal')
ax2.set_axis_off()
ax2.set_title('Tangent plane measured with the metric',loc='left',pad=18)
ax2.text(0,-.39, r'$\langle T,v_{\rm cool}\rangle_m<0$ : further cooling reduces distance',
         ha='center',fontsize=11)
fig.text(.04,.025,'Teal: arriving optimal-path direction.   Red: further-cooling direction.   Arrow lengths rescaled for readability.',fontsize=10)
fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.13,wspace=.19)
output=Path(__file__).resolve().parents[1]/'figures'/'qutrit-cooling'
fig.savefig(output/'qutrit-endpoint-vectors.png',dpi=200,bbox_inches='tight',pad_inches=.05)
fig.savefig(output/'qutrit-endpoint-vectors.svg',bbox_inches='tight',pad_inches=.05)
report=dict(xi=xi,D=D,metric_angle_degrees=float(angle),distance_derivative=float(derivative),
            metric_pairing_cooling=float(dot))
(output/'endpoint-vectors.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
