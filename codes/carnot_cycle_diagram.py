"""Schematic reversible Carnot cycle matching the manuscript's scaled Hamiltonians."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

navy,teal,red,amber='#19344A','#007E87','#C54E53','#CF8B24'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':navy,
 'axes.labelcolor':navy,'xtick.color':navy,'ytick.color':navy,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9.4,6.4))
a,b,tc,th=.3,.85,1,2
ax.add_patch(Rectangle((a,tc),b-a,th-tc,facecolor=amber,alpha=.12,edgecolor='none'))
ax.plot([a,b],[th,th],color=red,lw=3)
ax.plot([a,b],[tc,tc],color=teal,lw=3)
ax.plot([a,a],[tc,th],color=navy,lw=2)
ax.plot([b,b],[tc,th],color=navy,lw=2)
for start,end,color in [((.51,th),(.64,th),red),((b,1.65),(b,1.4),navy),((.64,tc),(.51,tc),teal),((a,1.4),(a,1.65),navy)]:
 ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',color=color,lw=2,mutation_scale=19))
for x,y in [(a,th),(b,th),(b,tc),(a,tc)]:ax.scatter(x,y,s=36,color=navy,zorder=5)
ax.text(a-.025,th+.055,r'$A:\ H$',ha='right',va='bottom')
ax.text(b+.025,th+.055,r'$B:\ \lambda H$',ha='left',va='bottom')
ax.text(b+.025,tc-.055,r'$C:\ \lambda\frac{T_c}{T_h}H$',ha='left',va='top')
ax.text(a-.025,tc-.055,r'$D:\ \frac{T_c}{T_h}H$',ha='right',va='top')
ax.text((a+b)/2,th+.13,'1  Hot isotherm',color=red,ha='center',weight='bold')
ax.text((a+b)/2,th+.30,r'$Q_h=T_h\Delta S$ enters',color=red,ha='center')
ax.text((a+b)/2,tc-.22,'3  Cold isotherm',color=teal,ha='center',weight='bold')
ax.text((a+b)/2,tc-.40,r'$|Q_c|=T_c\Delta S$ leaves',color=teal,ha='center')
ax.text(b+.025,1.5,'2  Isentropic\ncooling\n'+r'$Q=0$',ha='left',va='center',linespacing=1.5)
ax.text(a-.025,1.5,'4  Isentropic\nheating\n'+r'$Q=0$',ha='right',va='center',linespacing=1.5)
ax.text((a+b)/2,1.57,'Enclosed area = work out',ha='center',fontsize=12)
ax.text((a+b)/2,1.40,r'$W_{\rm rev}=(T_h-T_c)\Delta S$',ha='center',fontsize=14)
ax.set(xlim=(.01,1.13),ylim=(.42,2.55),xlabel='Entropy $S$',ylabel='Temperature $T$')
ax.set_xticks([a,b],labels=[r'$S_{\rm lo}$',r'$S_{\rm hi}$'])
ax.set_yticks([tc,th],labels=[r'$T_c$',r'$T_h$'])
ax.tick_params(length=0,pad=9)
for spine in ['top','right']:ax.spines[spine].set_visible(False)
for spine in ['bottom','left']:ax.spines[spine].set_color('#A5B1B9')
ax.set_title('Carnot cycle: fixed baths, driven Hamiltonian',loc='left',pad=15,fontsize=16)
fig.text(.5,.015,r'Reversible reference cycle; $0<\lambda<1$. Corner labels give the actual Hamiltonian.',ha='center',fontsize=10)
fig.subplots_adjust(left=.09,right=.97,bottom=.13,top=.89)
output=Path(__file__).resolve().parents[1]/'figures/thermal-machines'
output.mkdir(parents=True, exist_ok=True)
for suffix in ['png','svg','pdf']:
 fig.savefig(output/f'carnot-temperature-entropy.{suffix}',dpi=200,bbox_inches='tight',pad_inches=.06)
