import sys
sys.path.append('F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')
sys.path.append('F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')
import numpy as np
import matplotlib.pyplot as plt
from MDKernelEstimator.Estimator import DistanceTrainer
from CCT_10m39b.DataAnalysis.ExtractData import X_train,Y_train
import os

# %%
# Linear Regression

# reg,Yp = compare_real_predict(Xtrain,Ytrain,Xtest,Ytest,LinearRegression())
# reg,Yp = compare_real_predict(Xtrain,Ytrain,Xtest,Ytest,RidgeCV())
# V = np.corrcoef(X.T)
# reg = KNeighborsRegressor(n_neighbors=10,weights='distance',metric='mahalanobis',metric_params={'V':V})

import pickle
from CCT_10m39b.Simulation.plotfuncs import mysavefig


change_label = True

ACCs_file = os.path.join('..','RESULTS','ACCs.dat')
if os.path.isfile(ACCs_file):
    with open(ACCs_file,'rb') as f:
        ACCs = pickle.load(f)
else:
    ACCs = {}
    BSs = 2**np.arange(2,11)
    for bs in BSs:
        if bs in ACCs.keys():
            continue
        reg = DistanceTrainer(batch_size=bs, epochs=10000,Tmax=100)
        reg.fit(X_train, Y_train, rec_acc=True)

        accs = reg.ph.accuracies
        ACCs[bs] = accs
        with open(ACCs_file,'wb') as f:
            pickle.dump(ACCs.copy(),f)
        print(str(bs)+' finished')


#%%
if change_label:
    title_t = 'Training-efficiency-batchsize'
    bssym = 'batch-size'
    xylabel = ['训练时间 (s)', 'MAR']
    xlabelfont = 'Simhei'
    offset = 0.0035
    offset_lim = 0.002
    yticks = np.linspace(0.975,0.995,5)
    ylimt = [np.min(yticks),0.995]
else:
    title_t = 'training efficiency'
    bssym = 'batch-size'
    xylabel = ['training time (sec)', 'accuracy']
    xlabelfont = None
    offset = 0
    offset_lim = 0
    yticks = np.linspace(0.97,0.99,5)
    ylimt = [np.min(yticks),np.max(yticks)]

figt = plt.figure(title_t)
figt.clf()
axt = figt.gca()

lst = []
legends = []
for bs in ACCs.keys():
    if bs>=128:
        xt = np.array(list(ACCs[bs].keys()))
        yt = np.array(list(ACCs[bs].values())) + offset
        lt, = axt.plot(xt, yt, linewidth=0.125+0.0125*(1.6**np.log2(bs)))
        lst.append(lt)
        legends.append(('%s=%d' % (bssym, bs)))


yticklabels = ['%.1f%%' % (t*100) for t in yticks]

axt.set_yticks(yticks)
axt.set_yticklabels(yticklabels)
axt.set_ylim(ylimt)
axt.set_xlim([0,50])
axt.set_position((0.12,0.22,0.86,0.75))

if xlabelfont is not None:
    axt.set_xlabel(xylabel[0],fontname=xlabelfont)
else:
    axt.set_xlabel(xylabel[0])
axt.set_ylabel(xylabel[1])

axt.legend(lst,legends,loc='lower center',bbox_to_anchor=(0,-0.3,1,0.8),ncol=3)
figt.show()

if change_label:
    mysavefig(figt,title_t)



