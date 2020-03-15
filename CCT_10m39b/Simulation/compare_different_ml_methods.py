import sys
sys.path.append('F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')
sys.path.append('F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')

# %%
import os
import numpy as np
import scipy.io as scio

from MDKernelEstimator.Estimator import Estimator
from CCT_10m39b.DataAnalysis.ExtractData import X_train, Y_train, X_test, Y_test
from CCT_10m39b.Simulation.plotfuncs import plot_fore,plot_acc_vs_con,plot_err_multi_model,plot_ETEST_mean
from CCT_10m39b.Simulation.plotfuncs import plot_ETEST_err,plot_err_hist,plot_err_hist_curve,plot_err_hist_curve_3d
from CCT_10m39b.Simulation.plotfuncs import plot_acc,plot_err_rate,plot_pdf_3d,mesh_cdf_3d
from CCT_10m39b.Simulation.get_data_funcs import get_mat_pdf, get_CCTs, vec_confidence, get_errs_multi_model
from CCT_10m39b.Simulation.get_data_funcs import get_data_ETEST_r, get_ETEST_new, get_ETMe,get_bins_data


# %%
np.random.seed(66)
pos_all = [0.12,0.12,0.85,0.85]

# %%
PATH_OUTPUT = os.path.join('..','RESULTS', 'CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate')
kernel_model_para = scio.loadmat(os.path.join(PATH_OUTPUT,'ml.mat'))

Ntest = Y_test.size
Ntrain = Y_train.size
N = Ntest + Ntrain
Ppicks = np.linspace(0.2,1,20)
Npicks = [int(x) for x in np.ceil(Ppicks*Ntrain)]


# %%
estimator = Estimator(X_train, Y_train, kernel_model_para['KN'][0][0], kernel_model_para['m'])

# %%
if __name__=='__main__':
    change_label=True
    mat_pdf, mat_cdf, vec_y_pdf = get_mat_pdf()
    idx_sort = np.argsort(Y_test)
    mat_pdf_s = mat_pdf[idx_sort, :]
    mat_cdf_s = mat_cdf[idx_sort, 1:]
    azim1,azim2,elev1,elev2,exXY = 60,60,30,30,False
    
    # %%
    mesh_cdf_3d(1-mat_cdf_s,vec_y_pdf,pdf=False,azim=azim1,elev=elev1,save=change_label,exXY=exXY)
    mesh_cdf_3d(mat_pdf_s,vec_y_pdf,cmap=None,pdf=True,azim=azim2,elev=elev2,save=change_label,exXY=exXY)
    CCTs,dict_ccts_conf,errs_conf,conservs_conf,Ytest_new,Y_predict = get_CCTs(mat_cdf, vec_y_pdf)

    # %%
    plot_fore(CCTs,Ytest_new,Y_predict,vec_confidence,change_label=change_label)

    # %%
    fig_acc_cons = plot_acc_vs_con(vec_confidence,errs_conf,conservs_conf,change_label=change_label)

    # %%
    ETT = get_errs_multi_model()
    ETEST = ETT['test']

    # %%
    # plot_err_multi_model(ETT,Npicks,N)

    # %%
    ETEST, ETEST_mean = get_data_ETEST_r(ETEST)
    plot_ETEST_mean(ETEST_mean,Npicks,N,change_label=change_label)

    # %%
    dens_err = np.linspace(0, 1, Ntest)
    ETEST_new = get_ETEST_new(ETEST, ETEST_mean)
    # plot_ETEST_err(ETEST_new,dens_err)

    # %%
    ETMe = get_ETMe(ETEST_new)

    # %%
    methods = list(ETEST_new.keys())
    Nbins = 101
    ns,binss,patches = plot_err_hist(methods,ETMe,Nbins)
    idx_proposed = methods.index('Proposed')

    Ns,freqs,Xbinss, XbinssM = get_bins_data(ns,binss,len(methods),ETMe.shape[1])
    plot_err_hist_curve(methods, idx_proposed, freqs, binss, Xbinss, change_label=change_label)
    IDX_a = np.repeat(np.reshape(np.arange(len(methods)), (-1, 1)), freqs.shape[1], axis=1)

    # plot_err_hist_curve_3d(methods,IDX_a,XbinssM,freqs,binss)


    # %%
    ACCTEST_new = {k: 1 - np.abs(v[-1, :] / Y_test) for k, v in ETEST_new.items()}
    # plot_acc(ACCTEST_new,dens_err)

    # %%
    ERRRATETEST_new = {k: v[-1, :] / Y_test for k, v in ETEST_new.items()}
    # plot_err_rate(ERRRATETEST_new,dens_err)

    # %%
    MAE = {k:np.mean(np.abs(v[-1,:]))*1000 for k,v in ETEST_new.items()}
    RMSE = {k:np.sqrt(np.mean(np.square(v[-1,:])))*1000 for k,v in ETEST_new.items()}
    MER = {k:np.mean(np.abs(v)) for k,v in ERRRATETEST_new.items()}
    MAR = {k:np.mean(v) for k,v in ACCTEST_new.items()}

    I = []
    keys_sorted = ['Proposed','ANN','LASSO','BR','SVR','CART']
    for T in [MAE,RMSE,MER,MAR]:
        t = [T[k] for k in keys_sorted if k in T.keys()]
        I.append(t)

    I = np.array(I).T

    # %%
    print('end')
