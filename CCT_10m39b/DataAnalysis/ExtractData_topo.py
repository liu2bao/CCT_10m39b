import sys

sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\PyPSASP')
from PyPSASP.utils import utils_sqlite
from PyPSASP.PSASPClasses import LoadFlow
from PyPSASP.PSASPClasses.PSASP import CCT_KEY
from PyPSASP.constants import const
import numpy as np
from CCT_10m39b.utils.tools import formulate_LF_Matrix
import scipy.io as scio
import os

# %%
np.random.seed(66)

# %%
PATH_OUTPUT = os.path.join('..','RESULTS','CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate_topoChange')
dbt = const.RecordMasterDb
PATH_DB = os.path.join(PATH_OUTPUT, dbt)
datanamet, dbpostfix = os.path.splitext(dbt)
matpatht = os.path.join(PATH_OUTPUT, datanamet + '.mat')


# %%
def get_raw_data(path_db=PATH_DB, matpath=matpatht):
    if os.path.isfile(matpath):
        dt = scio.loadmat(matpath)
        XCC = dt['XCC']
        Y = dt['Y'].flatten()
        Topo = dt['Topo']
    else:
        data_raw = utils_sqlite.read_db(path_db, const.RecordMasterTable, return_dict_form=True)
        # data_raw = data_raw[0:100]

        # %%
        idx_t = [hh for hh in range(len(data_raw)) if const.LABEL_RESULTS in data_raw[hh][const.LABEL_LF].keys()]
        list_LF = [data_raw[hh][const.LABEL_LF][const.LABEL_RESULTS] for hh in idx_t]
        list_LFObj = [LoadFlow.LoadFlow(lf) for lf in list_LF]
        list_CCT = [data_raw[hh][CCT_KEY] for hh in idx_t]
        NT = len(list_CCT)
        flag_sel_CCT = np.array([t is not None for t in list_CCT])
        LFkeys, Xraw = formulate_LF_Matrix(list_LFObj)
        Traw = np.isnan(Xraw)

        Xraw[Traw] = 0
        Yraw = np.array(list_CCT)

        # %%
        LFkeys_ele = list(set([k[0:2] for k in LFkeys]))
        kidx = [None] * len(LFkeys)
        for kk in range(len(LFkeys)):
            lfk = LFkeys[kk]
            for hh in range(len(LFkeys_ele)):
                if lfk[0:2] == LFkeys_ele[hh]:
                    kidx[kk] = hh
                    break

        kidx = np.array(kidx)
        kidx_inverse = [np.where(kidx == kk)[0] for kk in range(len(LFkeys_ele))]
        topo_unchanged = []
        topo_inconsistent = []
        topo_not_uniform = []
        topo_changed = []
        for kk in range(len(LFkeys_ele)):
            idxt = np.where(kidx == kk)[0]
            Tt = Traw[:, idxt]
            pTt = np.unique(Tt, axis=0)
            if pTt.shape[0] == 1:
                topo_unchanged.append(kk)
            elif pTt.shape[0] == 2:
                if len(idxt) > 1 and (np.unique(pTt, axis=1)).shape[1] > 1:
                    topo_not_uniform.append(kk)
                else:
                    topo_changed.append(kk)
            else:
                topo_inconsistent.append(kk)
        topo_single = topo_changed + topo_not_uniform
        topo_multiple = topo_inconsistent
        idx_single = np.array([kidx_inverse[t][0] for t in topo_single])
        if topo_multiple:
            idx_multiple = np.hstack([kidx_inverse[t] for t in topo_multiple])
            idx_topo_all = np.hstack([idx_single, idx_multiple])
        else:
            idx_topo_all = topo_single
        Tpar_raw = Traw[:, idx_topo_all]

        # %%
        Vraw = np.corrcoef(Xraw.T)
        flag_sel_fea = ~np.all(np.isnan(Vraw), 0)

        X = Xraw[flag_sel_CCT, :]
        X = X[:, flag_sel_fea]
        # X = X[:,0:349]
        Y = Yraw[flag_sel_CCT].astype(float)
        Topo = Tpar_raw[flag_sel_CCT, :]

        XC = (X - np.min(X, 0)) / (np.max(X, 0) - np.min(X, 0))
        flag_sel_fea_2 = ~np.any(np.isnan(XC), axis=0)
        XCC = XC[:, flag_sel_fea_2]

        # %%
        if matpath is not None:
            scio.savemat(matpath, {'X': X, 'Y': Y, 'Topo': Topo,
                                    'flag_sel_CCT': flag_sel_CCT,
                                    'flag_sel_fea': flag_sel_fea,
                                    'flag_sel_fea_2': flag_sel_fea_2,
                                    'XC': XC,
                                    'XCC': XCC,
                                    'LFkeys': LFkeys,
                                    'idx_topo_all': idx_topo_all})
    return XCC, Topo, Y


def select_data(XCC, Topo, Y, cct_min=0.3, cct_max=0.6):
    idx_CCT = np.logical_and(Y>cct_min, Y<cct_max)
    Topo_s = Topo[idx_CCT,:]
    Y_s = Y[idx_CCT]
    XCC_s = XCC[idx_CCT,:]

    # ymax = np.max(Y)
    # print(ymax)
    # %%
    Topo_s = np.zeros(Topo_s.shape, dtype=np.int8)
    Topo_s[Topo_s==1] = 1
    Topo_s[Topo_s!=1] = -1
    XCC_s = (XCC_s - 0.5) * 2
    return XCC_s, Topo_s, Y_s


def extract_topo_data(cct_min=0.3, cct_max=0.6, path_db=PATH_DB, matpath=matpatht, return_topo=False):
    XCC, Topo, Y = get_raw_data(path_db=path_db, matpath=matpath)
    XCC_s, Topo_s, Y_s = select_data(XCC, Topo, Y, cct_min=cct_min, cct_max=cct_max)
    XTopo_s = np.hstack([XCC_s, Topo_s])
    # XTopo_s = XCC_s.copy()

    # %%
    # X = X[:200,:5]
    # Y = Y[:200]

    Nall = Y_s.size
    Ntrain = round(Nall * 0.75)
    idx_all = np.arange(0, Nall, dtype='int32')
    idx_all_permute = np.random.permutation(idx_all)
    idx_train = idx_all_permute[0:Ntrain]
    idx_test = idx_all_permute[Ntrain:]

    '''
    C = np.zeros([X.shape[1],1])
    for i in range(X.shape[1]):
        C[i] = np.corrcoef(X[:,i],Y)[0,1]
    '''

    if return_topo:
        T_u, I = np.unique(Topo_s, return_inverse=True, axis=0)
        X_train = XCC_s[idx_train, :]
        X_test = XCC_s[idx_test, :]
        T_train = Topo_s[idx_test, :]
        T_test = Topo_s[idx_test, :]
        I_train = I[idx_train]
        I_test = I[idx_test]
        Y_train = Y_s[idx_train]
        Y_test = Y_s[idx_test]

        return X_train, X_test, T_train, T_test, I_train, I_test, Y_train, Y_test
    else:
        X_train = XTopo_s[idx_train, :]
        X_test = XTopo_s[idx_test, :]
        Y_train = Y_s[idx_train]
        Y_test = Y_s[idx_test]

        return X_train, X_test, Y_train, Y_test