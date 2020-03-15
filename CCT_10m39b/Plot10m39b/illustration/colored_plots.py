import sys

sys.path.append('F:\Programs\PythonWorks\PowerSystem\CCT_10m39b_lxz')

# %%
from CCT_10m39b.Plot10m39b.plot10m39b import SLGdrawer, get_coordinates, plot_10m39b_basic
from CCT_10m39b.Plot10m39b.plot10m39b import KEY_TYPE, KEY_NO, KEY_COOR, KEY_COLOR, KEY_CONN, KEY_DIRE
from CCT_10m39b.Plot10m39b.plot10m39b import TYPE_LINE_AC, TYPE_LINE_TRANS,TYPE_SB_GEN,TYPE_SB_LOAD
from CCT_10m39b.Simulation.plotfuncs import mysavefig
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as col
import matplotlib.cm as cm
import scipy.io as scio
import os

# my_cmap_red = col.LinearSegmentedColormap.from_list('my_cmap', ['#FFD700', '#FF0000'])
my_cmap_red = col.LinearSegmentedColormap.from_list('my_cmap', ['#FFC1C1', '#FF0000'])
my_cmap_green = col.LinearSegmentedColormap.from_list('my_cmap', ['#E8FFC4', '#467500'])
my_cmap_blue = col.LinearSegmentedColormap.from_list('my_cmap', ['#9393FF', '#000079'])

buses, special_buses, lines = get_coordinates()

# %%
title_t = 'ILLUSTRATION-COLORED-SLG'
figsize = (12,5.5)
subtitles = ['有功子图','无功子图']

if __name__=='__main__':
    plt.close(title_t)
    figt = plt.figure(title_t,figsize=figsize)
    figt.clf()
    cmap_autumn = cm.autumn.reversed()
    #cmaps = [my_cmap_red, my_cmap_red]
    cmaps = [cmap_autumn]*2
    for i in range(2):
        cmap_t = cmaps[i]
        func_get_c_t = lambda: cmap_t(np.random.rand())
        if i==1:
            diff_color_line = True
        else:
            diff_color_line = False
        '''
        for k in buses.keys():
            buses[k][KEY_COLOR] = func_get_c_t()
        '''
        
        for k in range(len(special_buses)):
            type_t = special_buses[k][KEY_TYPE]
            if type_t==TYPE_SB_GEN:
                color_t = func_get_c_t()
            elif type_t==TYPE_SB_LOAD:
                color_t = func_get_c_t()
            else:
                color_t = None
            if color_t:
                special_buses[k][KEY_COLOR] = color_t
    
        for k in range(len(lines)):
            if diff_color_line:
                color_t = np.array([func_get_c_t() for ii in range(2)])
            else:
                color_t = func_get_c_t()
            lines[k][KEY_COLOR] = color_t
    
    
        # %%
        ax = figt.add_subplot(1,2,i+1)
        ax.set_xlim([80, 680])
        ax.set_ylim([20, 620])
        # ax.plot([0,500],[0,500],lw=10)
        Drawer = SLGdrawer(buses, special_buses, lines, ax)
        Drawer.spec_color = True
        Drawer.text_bus = True
        Drawer.plot_slg(scale=25)
        sc = cm.ScalarMappable(cmap=cmap_t)
        sc.set_array([1,0])
        #cax = figt.colorbar(sc,orientation='horizontal')
        cax = figt.colorbar(sc)
        #cax.set_position([0.05,0.05,0.9,0.05])
        ax.set_title(subtitles[i],fontname='Simhei')
    
    #%%
    figt.tight_layout(pad=1)
    figt.show()
    mysavefig(figt)
