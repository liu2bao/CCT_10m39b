import numpy as np
from CCT_10m39b.utils.tools import formulate_LF_Matrix
import scipy.io as scio
import os
import sys
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\PyPSASP')
from PyPSASP.utils import utils_sqlite
from PyPSASP.PSASPClasses import LoadFlow
from PyPSASP.PSASPClasses.PSASP import CCT_KEY
from PyPSASP.constants import const

np.random.seed(66)

#%%
path_t = os.path.abspath(__file__)
path_t, file_t = os.path.split(path_t)
path_t = os.path.abspath(os.path.join(path_t,'..'))
PATH_OUTPUT = os.path.join(path_t,'RESULTS', 'CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate')
dbt = const.RecordMasterDb
data_name_t, db_postfix = os.path.splitext(dbt)
mat_path_t = os.path.join(PATH_OUTPUT, data_name_t + '.mat')
mat_path_extracted = os.path.join(PATH_OUTPUT, data_name_t + '_extracted.mat')

if os.path.isfile(mat_path_extracted):
    data_t = scio.loadmat(mat_path_extracted)
    X_train, Y_train, X_test, Y_test = [data_t[k] for k in ['X_train', 'Y_train', 'X_test', 'Y_test']]
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()
    XCC = np.vstack([X_train, X_test])
    Y = np.append(Y_train, Y_test)
else:
    if os.path.isfile(mat_path_t):
        dt = scio.loadmat(mat_path_t)
        XCC = dt['XCC']
        Y = dt['Y'].flatten()
    else:
        # dbt = 'record_master_simple.db'
        db_path = os.path.join(PATH_OUTPUT, dbt)
        data_raw = utils_sqlite.read_db(db_path, const.RecordMasterTable, return_dict_form=True)
        # data_raw = data_raw[0:100]

        # %%
        list_LF = [d[const.LABEL_LF][const.LABEL_RESULTS] for d in data_raw]
        list_LFObj = [LoadFlow.LoadFlow(lf) for lf in list_LF]
        list_CCT = [d[CCT_KEY] for d in data_raw]
        NT = len(list_CCT)
        flag_sel_CCT = np.array([t is not None for t in list_CCT])
        LF_keys, X_raw = formulate_LF_Matrix(list_LFObj)
        Y_raw = np.array(list_CCT)

        # %%
        V_raw = np.corrcoef(X_raw.T)
        flag_sel_fea = ~np.all(np.isnan(V_raw), 0)

        X = X_raw[flag_sel_CCT, :]
        X = X[:, flag_sel_fea]
        # X = X[:,0:349]
        Y = Y_raw[flag_sel_CCT].astype(float)

        XC = (X - np.min(X, 0)) / (np.max(X, 0) - np.min(X, 0))
        flag_sel_fea_2 = ~np.any(np.isnan(XC), axis=0)
        XCC = XC[:, flag_sel_fea_2]

        scio.savemat(mat_path_t, {'X': X, 'Y': Y,
                                  'flag_sel_CCT': flag_sel_CCT,
                                  'flag_sel_fea': flag_sel_fea,
                                  'flag_sel_fea_2': flag_sel_fea_2,
                                  'XC': XC, 'XCC': XCC, 'LFkeys': LF_keys})
    N_all = Y.size
    N_train = round(N_all * 0.75)
    idx_all = np.arange(0, N_all, dtype='int32')
    idx_all_permute = np.random.permutation(idx_all)
    idx_train = idx_all_permute[0:N_train]
    idx_test = idx_all_permute[N_train:]

    '''
    C = np.zeros([X.shape[1],1])
    for i in range(X.shape[1]):
        C[i] = np.corrcoef(X[:,i],Y)[0,1]
    '''

    X_train = XCC[idx_train, :]
    X_test = XCC[idx_test, :]
    Y_train = Y[idx_train]
    Y_test = Y[idx_test]

    scio.savemat(mat_path_extracted, {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test})
