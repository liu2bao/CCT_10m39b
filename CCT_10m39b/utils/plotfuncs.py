# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:06:00 2019

@author: LiuXianzhuang
"""

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

# %%
# bluecolor = '#BBFFFF'
# redcolor = '#FF6A6A'
bluecolor = 'blue'
redcolor = 'red'
purplecolor = 'purple'
whitecolor = 'white'
my_cmap = col.LinearSegmentedColormap.from_list('my_cmap', [bluecolor, redcolor])
my_cmap_red = col.LinearSegmentedColormap.from_list('my_cmap_red', [redcolor, whitecolor])
my_cmap_blue = col.LinearSegmentedColormap.from_list('my_cmap_blue', [bluecolor, whitecolor])

my_font = {'size': 10}

h1 = 0.499
h2 = 0.501
cdict = {'blue': ((0.0, 1.0, 1.0),
                  (h1, 0.0, 0.0),
                  (h2, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),

         'green': ((0.0, 1.0, 1.0),
                   (h1, 0.0, 0.0),
                   (h2, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'red': ((0.0, 1.0, 1.0),
                 (h1, 1.0, 1.0),
                 (h2, 0.0, 0.0),
                 (1.0, 1.0, 1.0))
         }

cdict2 = {'blue': ((0.0, 0.0, 0.0),
                   (h1, 0.0, 0.0),
                   (h2, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (h1, 0.0, 0.0),
                    (h2, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'red': ((0.0, 1.0, 1.0),
                  (h1, 1.0, 1.0),
                  (h2, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),

          'alpha': ((0.0, 0.0, 0.0),
                    (h1, 1.0, 1.0),
                    (h2, 1.0, 1.0),
                    (1.0, 0.0, 0.0))
          }

pm = 0.2
am = 0.8
cdict3 = {'blue': ((0.0, 0.0, 0.0),
                   (pm, am, am),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (pm, am, am),
                    (0.5, 1.0, 1.0),
                    (1 - pm, am, am),
                    (1.0, 0.0, 0.0)),

          'red': ((0.0, 1.0, 1.0),
                  (0.5, 1.0, 1.0),
                  (1 - pm, am, am),
                  (1.0, 0.0, 0.0)),

          }

my_cmap_all = col.LinearSegmentedColormap('my_cmap_all', cdict3)
# my_cmap_all = cm.coolwarm

# extra arguments are N=256, gamma=1.0
cm.register_cmap(cmap=my_cmap)
cm.register_cmap(cmap=my_cmap_red)
cm.register_cmap(cmap=my_cmap_blue)

#%%

def mosaic(mat,colormap,ax):
    s1, s2 = mat.shape
    mat_r = (mat-np.min(mat))/(np.max(mat)-np.min(mat))
    for i in range(s1):
        for j in range(s2):
            mt = mat_r[i,j]
            ct = colormap(mt)
            rect = mpatches.Rectangle((i,j),width=1,height=1,fill=True,
                                      edgecolor='none',facecolor=ct)
            ax.add_patch(rect)
    ax.set_xlim([0,s1])
    ax.set_ylim([0,s2])

# %%

def plot_conf_curves(Ys, vec_conf, ax_fore, Xtest=None):
    Ntest = Ys.shape[1]
    if Xtest is None:
        Xtest = np.arange(Ntest)
    le = []
    ys_ori = np.zeros(Ntest)
    x_fill_di = Xtest
    count = 0
    for ct in vec_conf:
        ys_t = Ys[count, :]
        if count > 0:
            colt = my_cmap_all(ct)
            ax_fore.fill_between(x_fill_di, ys_ori, ys_t, facecolor=colt, edgecolor='none')
        le.append('%.3f' % ct)
        ys_ori = ys_t
        count += 1


def add_cbar_to_conf_curve(ax_fore, fig_fore):
    sm = plt.cm.ScalarMappable(cmap=my_cmap_all, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    ax_fore.set_position([0.12, 0.2, 0.84, 0.77])
    cax = fig_fore.add_axes([0.12, 0.1, 0.84, 0.03])
    cbar = fig_fore.colorbar(sm, ax=ax_fore, cax=cax, orientation='horizontal')
    xticks_cax = np.linspace(0, 1, 5)
    xticklabels_cax = [('%d%%' % (t * 100)) for t in xticks_cax]
    cbar.set_ticks(xticks_cax)
    cbar.set_ticklabels(xticklabels_cax)
    cax.set_xlabel('$\epsilon$', fontdict=my_font)
    ax_fore.set_ylabel(r'$y_{{\rm{CCT}},m}$ (s)', fontdict=my_font)
    ax_fore.set_xlabel('$m$')


def plot_median_mean_real(ax_fore,Xtest,Ytest,y_median,y_mean):
    l_real, = ax_fore.plot(Xtest, Ytest, lw=1.5, alpha=0.6, c='k')
    l_median, = ax_fore.plot(Xtest, y_median, lw=2, c='#00FF00', linestyle='--')
    l_mean, = ax_fore.plot(Xtest, y_mean, lw=2, c='#FFFF00', linestyle='-.')
    ax_fore.legend([l_real, l_median, l_mean],
                   [r'$\overline{y}_{{\rm{CCT}},m}$',
                    r'$\hat{y}_{{\rm{CCT}},m}^{\rm{con}}(50\%)$',
                    r'$\hat{y}_{{\rm{CCT}},m}$'],
                   prop=my_font,
                   loc='upper left')

    minY = np.percentile(Ytest, 0)
    maxY = np.percentile(Ytest, 100)

    ax_fore.set_xlim([np.min(Xtest), np.max(Xtest)])
    ax_fore.set_ylim([minY - 0.1, maxY + 0.1])





#%%
if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    import scipy.io as scio
    
    # %%
    PATH_OUTPUT = os.path.join('RESULTS','CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate_topoChange')
    dbt = 'record_master.db'
    datanamet, dbpostfix = os.path.splitext(dbt)
    matpatht = os.path.join(PATH_OUTPUT, datanamet + '.mat')
    dt = scio.loadmat(matpatht)
    XCC = dt['XCC']
    Y = dt['Y'].flatten()
    Topo = dt['Topo']
    Topo_s = np.zeros(Topo.shape)
    Topo_s[Topo==1] = 1
    Topo_s[Topo!=1] = -1
    XCC_s = (XCC - 0.5) * 2
    XTopo_s = np.hstack([XCC_s, Topo_s])
    idxt = np.argsort(Y)
    XTopo_s_sorted = XTopo_s[idxt,:]
    mat = XTopo_s_sorted[0::50,:]
    figt = plt.figure('t')
    figt.clf()
    axt = figt.gca()
    mosaic(mat,cm.coolwarm,axt)
            
