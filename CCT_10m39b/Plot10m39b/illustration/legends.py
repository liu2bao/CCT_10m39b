import sys
sys.path.append('F:\Programs\PythonWorks\PowerSystem\CCT_10m39b_lxz')

# %%
from CCT_10m39b.Plot10m39b.plot10m39b import SLGdrawer,get_coordinates,plot_10m39b_basic
from CCT_10m39b.Plot10m39b.plotSMIB import plot_SMIB_basic
from CCT_10m39b.Simulation.plotfuncs import mysavefig
import matplotlib.pyplot as plt
import numpy as np

# %%
buses, special_buses, lines = get_coordinates()
titlet = 'SLG-IEEE10M39B'
plt.close(titlet)
fig = plt.figure(titlet,figsize=(8,6))
fig.clf()
Drawer = plot_10m39b_basic(fig=fig,return_drawer=True,scale=25)
Drawer.plot_legend(offset=[40,0])
ax = fig.gca()
ax.set_xlim([80,880])
ax.set_ylim([20,620])
fig.show()
fig.tight_layout()

mysavefig(fig,postfix='.pdf')

# %%
titlet = 'SLG-SMIB'
plt.close(titlet)
fig = plt.figure(titlet,figsize=(7,3))
fig.clf()
Drawer = plot_SMIB_basic(fig=fig,return_drawer=True)
Drawer.plot_legend(loc='center lower',orientation='horizontal',scale_amp=0.32)

ax = fig.gca()
#ax.set_xlim([80,880])
ax.set_ylim([-70,50])
#fig.show()
fig.tight_layout()

mysavefig(fig,postfix='.pdf')
