import sys

sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')


import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from CCT_10m39b.Simulation.Topochange.MDlearn import mat_all_result_r,mat_train_test,NWKRA,NWKRB,NWKRAp,Y_test_r,flag_cont,idx_test_r,X_test_r
from CCT_10m39b.Simulation.plotfuncs import mapping_color,mysavefig
from CCT_10m39b.Simulation.gadgets import get_path_file_tables
import os

#%%
tex_name_table = 'erroridx-sta-tc'
path_tex_table = os.path.join(get_path_file_tables(),tex_name_table+'.tex')
data_train_test = scio.loadmat(mat_train_test)
data_results = scio.loadmat(mat_all_result_r)
# Y_test = data_train_test['Y_test'].flatten()
meth_show_order = [NWKRAp, NWKRA, NWKRB, 'ANN', 'LASSO', 'BLR', 'CART']
mapping_color[NWKRAp] = mapping_color['Proposed']
mapping_color[NWKRA] = 'DarkTurquoise'
mapping_color[NWKRB] = 'cyan'
mapping_color['BLR'] = mapping_color['BR']
lws = {NWKRA:1.5,NWKRB:1.5,NWKRAp:2}
linestyles = {NWKRA:'--',NWKRB:'--',NWKRAp:'-'}
markers = {}
lw_default = 1
linestyle_default = '-'
marker_default = 's'
# markersizes = {NWKRA:1.5,NWKRB:1.5,NWKRAp:2}
# markersize_default = 1

#%%
MEkey = r'\MEsym (ms)'
RMSEkey = r'\RMSEsym (ms)'
MERkey = r'\MERsym'
MARkey = r'\MARsym'
keys_indices = [MEkey, RMSEkey, MERkey, MARkey]

num_meth = len(meth_show_order)
num_indices = len(keys_indices)
INDVALS = np.zeros([num_meth, num_indices])
Ntest = len(Y_test_r)
ERRs = np.zeros([num_meth, Ntest])

#%%
for i in range(num_meth):
    meth = meth_show_order[i]
    yvec = data_results[meth]
    yvec = yvec.flatten()
    ERRt = (yvec-Y_test_r)*1000
    ERRs[i,:] = ERRt
    INDVALS[i][0] = np.mean(np.abs(ERRt))
    INDVALS[i][1] = np.sqrt(np.mean(np.square(ERRt)))
    INDVALS[i][2] = np.mean(np.abs(ERRt/(Y_test_r*1000)))
    INDVALS[i][3] = 1-INDVALS[i][2]

# %%
XBINS = np.linspace(-25,25,30)
DX = np.mean(np.diff(XBINS))
num_bins = len(XBINS)-1
XBINCS = [(XBINS[i]+XBINS[i+1])/2 for i in range(num_bins)]
HIST = np.zeros([num_meth, num_bins])
for i in range(num_meth):
    for j in range(num_bins):
        Xu = XBINS[j+1]
        Xl = XBINS[j]
        HIST[i,j] = len(np.where(np.logical_and(ERRs[i]>=Xl,ERRs[i]<Xu))[0])/DX/Ntest


# %%
figt = plt.figure('ERR-HIST-CURVE-TOPOCHANGE')
figt.clf()
axt = figt.gca()
ls = []
for i in range(num_meth):
    meth = meth_show_order[i]
    color_t = mapping_color[meth]
    lwt, linestyle_t, marker_t = lw_default, linestyle_default, marker_default
    if meth in lws.keys():
        lwt = lws[meth]
    if meth in linestyles.keys():
        linestyle_t = linestyles[meth]
    if meth in markers.keys():
        marker_t = markers[meth]
    lt, = axt.plot(XBINCS,HIST[i],c=color_t,lw=lwt,linestyle=linestyle_t,
                   marker=marker_t,markersize=lwt*2)
    ls.append(lt)

ylimt = axt.get_ylim()
axt.plot([0, 0], ylimt, c='k', linestyle='--', alpha=0.5)
axt.set_xlabel(r'${\rm{E}}_j$ (ms)')
axt.set_ylabel(r'频率密度 (${\rm{ms}}^{-1}$)',fontname='Simhei')
axt.legend(ls, meth_show_order, loc='upper right', bbox_to_anchor=(0.75, 0.6, 0.2, 0.3))
axt.set_ylim(ylimt)
axt.set_xlim([np.min(XBINCS),np.max(XBINCS)])
figt.tight_layout()
mysavefig(figt)

# %%
alignsym = ' & '
endsym = r' \\ '
hlinesym = r' \hline '
str_table_title = alignsym.join(['算法']+keys_indices)
amps = [1,1,100,100]
formats = ['%.2f','%.2f',r'\myprc[%.2f]',r'\myprc[%.2f]']
strs_indices = [(alignsym.join([meth]+[(ft % (indt*ampt)) for ft,indt,ampt in zip(formats,indval,amps)])) for meth,indval in zip(meth_show_order,INDVALS)]

strs_all = str_table_title+endsym+hlinesym+'\n'+((endsym+'\n').join(strs_indices)+endsym+'\n')

with open(path_tex_table,'wb') as f:
    f.write(strs_all.encode(encoding='UTF-8'))


#%%
prcs_show = np.linspace(1,99,5)
title_t = 'ERR-DISTRIBUTION-TOPO'
fig_err_distr = plt.figure(title_t)
fig_err_distr.clf()
ax_t = fig_err_distr.gca()
Ts_test = X_test_r[:,~flag_cont].astype(np.uint8)
Ts_test_u, idx_u, idx_q = np.unique(Ts_test, axis=0, return_index=True, return_inverse=True)
errmeth_show = [NWKRA,NWKRAp]

myERRs = [np.abs(ERRs[meth_show_order.index(k),:]) for k in errmeth_show]

myERRs_topo = [[err[idx_q==i] for i in range(len(idx_u))] for err in myERRs]
myERRs_topo_prcs = np.array([[[np.percentile(errt, prct) for prct in prcs_show] for errt in errtopo] for errtopo in myERRs_topo])

#err_tosort = np.squeeze(np.mean(myERRs_topo_prcs[:,:,int((len(prcs_show)+1)/2)],axis=0))
err_tosort = np.squeeze(myERRs_topo_prcs[1,:,int((len(prcs_show)+1)/2)])
idx_sort = np.argsort(err_tosort)

ax_t.plot(np.squeeze(myERRs_topo_prcs[1,idx_sort,:]))

#%%
title_t = 'ERR-COMPARISON-TOPO'
fig_err_compare = plt.figure(title_t)
fig_err_compare.clf()
ax_t = fig_err_compare.gca()
ax_t.plot([0,20],[0,20])
ax_t.scatter(myERRs[0],myERRs[1],s=0.1,c='orange')
# ax_t.set_xlim([0,16])
# ax_t.set_ylim([0,16])
