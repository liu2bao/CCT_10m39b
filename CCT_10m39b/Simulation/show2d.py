import sys
sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\PyPSASP')
from PyPSASP.utils import utils_sqlite
from PyPSASP.PSASPClasses.PSASP import CCT_KEY
from PyPSASP.constants import const
from PyPSASP.PSASPClasses import LoadFlow
import os
import numpy as np
import matplotlib.pyplot as plt
from CCT_10m39b.utils import compare_real_predict

from sklearn.linear_model import RidgeCV

#%%
PATH_OUTPUT = r'CCT_RESULTS_simple_4'

dbt = const.RecordMasterDb
#dbt = 'record_master_simple.db'
db_path = os.path.join(PATH_OUTPUT,dbt)
data_raw = utils_sqlite.read_db(db_path,const.RecordMasterTable,return_dict_form=True)
#data_raw = data_raw[0:100]

#%%
list_LF = [d[const.LABEL_LF][const.LABEL_RESULTS] for d in data_raw]
list_LFObj = [LoadFlow.LoadFlow(lf) for lf in list_LF]
list_LFexpanded = [list(lfobj.dict_lf_expanded.values()) for lfobj in list_LFObj]
Xraw = np.array(list_LFexpanded)
S=np.ptp(Xraw,axis=0)

list_G = [l[const.LABEL_GENERATOR] for l in list_LF]
list_G1 = [g[1][const.GenPgKey] for g in list_G]
list_G2 = [g[-3][const.GenPgKey] for g in list_G]

#X = np.vstack([list_G1,list_G2]).T
X = Xraw[:,S>0.01]
Y = np.array([d[CCT_KEY] for d in data_raw])

Nall = Y.size
Ntrain = round(Nall*0.75)
idx_all = np.arange(0,Nall,dtype='int32')
idx_all_permute = np.random.permutation(idx_all)
idx_train = idx_all_permute[0:Ntrain]
idx_test = idx_all_permute[Ntrain:]

C = np.zeros([X.shape[1],1])
for i in range(X.shape[1]):
    C[i] = np.corrcoef(X[:,i],Y)[0,1]
XC = (X-np.min(X,0))/(np.max(X,0)-np.min(X,0))

Xtrain = XC[idx_train,:]
Xtest = XC[idx_test,:]
Ytrain = Y[idx_train]
Ytest = Y[idx_test]

#%%
figt = plt.figure('3d')
figt.clf()
axt = figt.add_subplot(111,projection='3d')
axt.scatter(X[:,0],X[:,1],Y)

#%%
#regt = NN_model()
regt = RidgeCV()
#regt = LassoCV()
#regt = LinearRegression()
#regt = KNeighborsRegressor(n_neighbors=5)
#regt = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(64,), random_state=1)
reg,Yptest = compare_real_predict(Xtrain,Ytrain,Xtest,Ytest,regt)

Yptrain = reg.predict(Xtrain)
figt = plt.figure('train')
figt.clf()
axt = figt.gca()
idxt = np.argsort(Ytrain)
axt.plot(Yptrain[idxt],lw=0.5,c='r')
axt.plot(Ytrain[idxt],lw=0.5,c='b')

Etest = np.mean(np.square(Ytest-Yptest))
Etrain = np.mean(np.square(Ytrain-Yptrain))

