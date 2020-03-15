# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 03:40:32 2020

@author: LiuXianzhuang
"""

import sys

sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\PyPSASP')

#%%
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from CCT_10m39b.Simulation.plotfuncs import mapping_color,mysavefig
from CCT_10m39b.Simulation.gadgets import get_path_file_tables
from CCT_10m39b.Plot10m39b.plot10m39b import get_coordinates,SLGdrawer,KEY_COOR,KEY_COLOR,KEY_CONN
np.random.seed(66)

#%%
cutset_no = [[[2,25],[26,27]],[[2,25],[17,27]]]
title_t = 'Cutsets-10m39b'
plt.close(title_t)
fig_t = plt.figure(title_t,figsize=(8,4))
fig_t.clf()
nC = len(cutset_no)
buses, special_buses, lines = get_coordinates()
colors = ['orange','cyan']
label_cs = ['I','II']
for i in range(nC):
    csnt = cutset_no[i]
    ax_t = fig_t.add_subplot(1,nC,i+1)
    slgt = SLGdrawer(buses, special_buses, lines, ax_t)
    slgt.plot_slg()
    KK = []
    for K in csnt:
        coort = (buses[K[0]][KEY_COOR]+buses[K[1]][KEY_COOR])/2
        #ax_t.scatter(buses[K[0]][KEY_COOR][0],buses[K[0]][KEY_COOR][1])
        KK.append(coort)
    KK = np.array(KK)
    lt, = ax_t.plot(KK[:,0],KK[:,1],linestyle='--',marker='v'
              ,c=colors[i],linewidth=2)
    ax_t.legend([lt],[('输电断面%s' % label_cs[i])],
                prop={'family':'SimHei'},
                loc = 'right center')
        
fig_t.tight_layout()
mysavefig(fig_t)
    


