# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:52:40 2020

@author: LiuXianzhuang
"""

import sys
sys.path.append('..')
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')

# %%
import os
import numpy as np
import scipy.io as scio
import pickle

from MDKernelEstimator.Estimator import Estimator
from CCT_10m39b.Simulation.plotfuncs import mysavefig
import matplotlib.pyplot as plt

# %%
np.random.seed(66)
pos_all = [0.12,0.12,0.85,0.85]

# %%
PATH_OUTPUT = os.path.join('..','..','RESULTS', 'CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate')
mat_data = os.path.join(PATH_OUTPUT, 'record_master.mat')
data = scio.loadmat(mat_data)
X = data['XCC']
Y = data['Y'].flatten()
kernel_model_para = scio.loadmat(os.path.join(PATH_OUTPUT,'ml.mat'))

N = Y.size
train_prop = 0.8
test_prop = 0.2
H_W = 0.012
Ntrain = int(N*train_prop)
Ntest = int(N*train_prop)

Xtest = X[Ntrain:min(N,Ntrain+Ntest), :]
Ytest = Y[Ntrain:min(N,Ntrain+Ntest)]
Xtrain = X[:Ntrain, :]
Ytrain = Y[:Ntrain]

# %%
estimator = Estimator(Xtrain,Ytrain,kernel_model_para['KN'][0][0],kernel_model_para['m'])


def get_ents():
    file_ents = 'ents.dat'
    if os.path.isfile(file_ents):
        with open(file_ents,'rb') as f:
            vec_ents = pickle.load(f)
    else:
        Y_predict = estimator.predict(Xtest)
        yt, Pt = estimator.get_pdf(h_w=H_W)
        dy = np.mean(np.diff(yt))
        vec_ents = np.sum(-Pt*np.log(Pt),axis=1)*dy
        with open(file_ents,'wb') as f:
            pickle.dump(vec_ents,f)
    return vec_ents


def check_all_test_sets(n=10):
    N = n**2
    Nplot = int(np.ceil(Ntest/N))
    for i in range(Nplot):
        idx_t = np.arange(i*N,min(Ntest,(i+1)*N))
        Xtest_t = Xtest[idx_t,:]
        Y_predict = estimator.predict(Xtest_t)
        yt, Pt = estimator.get_pdf(h_w=H_W)
        figt = plt.figure(i)
        figt.clf()
        for j in range(N):
            axt = figt.add_subplot(n,n,j+1)
            pt = Pt[j, :]
            idx_sel_t = pt>=np.max(pt)*0.03
            ytt = yt[idx_sel_t]
            ptt = pt[idx_sel_t]
            axt.plot(ytt, ptt)
            axt.set_title(idx_t[j])
        figt.show()
        f = input()


def get_min_max_ent_idx(vec_ents):
    min_ent_idx = np.argmin(vec_ents)
    max_ent_idx = np.argmax(vec_ents)
    return min_ent_idx,max_ent_idx


def plot_PDF(min_ent_idx,max_ent_idx):
    figt = plt.figure('pdf_sel')
    figt.clf()
    idx_sel = [min_ent_idx,max_ent_idx]
    count_t = 1
    start_b = 0.08
    width_b = 0.4
    for idxt in idx_sel:
        axt = figt.add_subplot(1,len(idx_sel),count_t)
        x_test_t = Xtest[idxt, :]
        y_predict_t = estimator.predict(x_test_t)
        yt, pt = estimator.get_pdf(h_w=H_W)
        # yt,pt = WTM.predict_pdf(xtest_t,h_w=0.01)
        pt = pt.flatten()
        ft = pt > 0.05
        axt.plot(yt[ft], pt[ft], lw=2, c='r')
        axt.set_xlabel(r'$y_{{\rm{CCT}},%d}$ (s)' % count_t)
        axt.set_ylabel(r'PDF (s$^{-1}$)')
        axt.set_title('PDF$_%d$' % count_t)
        axt.set_position([start_b + (count_t - 1) / len(idx_sel), 0.15, width_b, 0.75])
        count_t += 1



def make_pdf_double(h_w=H_W):
    y1 = 0.4125
    y2 = 0.45
    p1 = 5/12.3
    p2 = 1-p1
    yt = np.linspace(0.35,0.5,500)
    pt_func = lambda y:1 / np.sqrt(2 * np.pi) / h_w * np.exp(-np.square(y-yt) / (2 * h_w * h_w))
    pt1 = pt_func(y1)
    pt2 = pt_func(y2)
    pt = p1*pt1+p2*pt2
    return yt,pt


def make_pdf_single(h_w=H_W*1.6):
    ys = 0.35
    yt = np.linspace(0.1,0.5,500)
    pt_func = lambda y:1 / np.sqrt(2 * np.pi) / h_w * np.exp(-np.square(y-yt) / (2 * h_w * h_w))
    pt = pt_func(ys)
    return yt,pt


def plot_pdf_r(plot_mean=True,ch=False,save=False):
    if ch:
        legends=['PDF','期望值']
    else:
        legends=['PDF','expectation']
    ypts = make_pdf_single()
    yptd = make_pdf_double()
    YPs = [ypts,yptd]
    title1 = 'PDF-examples'
    title2 = 'Ycon-examples'
    figsizet = (8,4)
    plt.close(title1)
    plt.close(title2)
    figt = plt.figure(title1,figsize=figsizet)
    figt.clf()
    figtt = plt.figure(title2,figsize=figsizet)
    figtt.clf()
    count_t = 1
    start_b = 0.08
    width_b = 0.4
    Np = len(YPs)
    for i in range(Np):
        ytt, ptt = YPs[i]
        dy = np.mean(np.diff(ytt))
        cdft = np.cumsum(ptt)*dy
        axt = figt.add_subplot(1,Np,count_t)
        axtt = figtt.add_subplot(1,Np,count_t)
        ftt = ptt > 0.05
        axt.plot(ytt[ftt], ptt[ftt], lw=2, c='r')
        axt.set_xlabel(r'$y$ (s)')
        axt.set_ylabel('PDF')
        axt.set_title(r'${\rm PDF}\left(x_%d, y\right)$' % count_t)
        axt.set_position([start_b + (count_t - 1) / Np, 0.15, width_b, 0.75])
        if plot_mean:
            xlimt = axt.get_xlim()
            ylimt = axt.get_ylim()
            ymean = np.sum(ytt*ptt*dy)/np.sum(ptt*dy)
            axt.plot([ymean,ymean],ylimt,c='k',linestyle='--',linewidth=0.5)
            axt.set_xlim(xlimt)
            axt.set_ylim(ylimt)
            axt.legend(legends,loc='upper left',prop={'family':'Simhei'})
            
        axtt.plot(1-cdft[ftt],ytt[ftt], lw=2, c='b')
        axtt.set_xlabel(r'$\sigma$')
        axtt.set_ylabel('CCT (s)')
        axtt.set_title(r'$\hat{Y}^{\rm c}(x_%d, \sigma)$' % count_t)
        axtt.set_position([start_b + (count_t - 1) / Np, 0.15, width_b, 0.75])
        xticks = np.linspace(0,1,6)
        xticklabels = [('%d%%' % (xt*100)) for xt in xticks]
        axtt.set_xticks(xticks)
        axtt.set_xticklabels(xticklabels)
        count_t += 1
    
    for fig in [figt, figtt]:
        #fig.tight_layout()
        fig.show()
        if save:
            mysavefig(fig)


if __name__=='__main__':
    plot_pdf_r(ch=True,save=True)

