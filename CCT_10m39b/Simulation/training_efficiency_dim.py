import sys

sys.path.append('F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')
sys.path.append('F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')
import numpy as np
import matplotlib.pyplot as plt
from MDKernelEstimator.Estimator import DistanceTrainer
from CCT_10m39b.DataAnalysis.ExtractData import X_train,Y_train
import os
from sklearn.linear_model import LinearRegression

#%%
import pickle
from CCT_10m39b.Simulation.plotfuncs import mysavefig

change_label = True
ctt = 'b'
cacc = 'r'

Nepoch = 100
MCs_file = os.path.join('..','RESULTS','MCs.dat')
if os.path.isfile(MCs_file):
    with open(MCs_file, 'rb') as f:
        MCs = pickle.load(f)
else:
    MCs = {}

    #%%
    Amps = np.hstack([1,np.arange(5,101,5)])
    for amp in Amps:
        if amp in MCs.keys():
            continue
        reg = DistanceTrainer(batch_size=1024, epochs=Nepoch+1, Tmax=500)
        Xtrain_t = np.hstack([X_train + np.random.rand(*X_train.shape) * 0.1 for i in range(amp + 1)])
        reg.fit(Xtrain_t, Y_train, rec_acc=True)

        accs = reg.ph.accuracies
        MCs[amp] = accs
        with open(MCs_file, 'wb') as f:
            pickle.dump(MCs.copy(), f)
        print(str(amp) + ' finished')

#%%

if change_label:
    titlet = 'Training-efficiency-dim'
else:
    titlet = 'training efficiency on dimension'

#%%
lw_ax = 1
p = 0.15
P = [p, p, 1 - 2 * p, 1 - 2 * p]
figt = plt.figure(titlet)
figt.clf()
axt = figt.gca()
ax_acc = axt.twinx()

axt.spines['left'].set_color(ctt)
ax_acc.spines['right'].set_color(cacc)
axt.spines['right'].set_visible(False)
ax_acc.spines['left'].set_visible(False)
ax_acc.set_position(P)
axt.set_position(P)
ax_acc.spines['left'].set_linewidth(lw_ax)
axt.spines['right'].set_linewidth(lw_ax)
ax_acc.spines['bottom'].set_linewidth(lw_ax)
ax_acc.spines['top'].set_visible(False)
axt.spines['top'].set_visible(False)

TT_dict = {k:list(v) for k,v in MCs.items() if k<=20}
ACC_dict = {k:list(v.values()) for k,v in MCs.items() if k<=20}
dimensions = np.array(list(TT_dict.keys()))*500
training_times = np.array([v[Nepoch-1]-v[0] if len(v)>=Nepoch else v[-1]/len(v)*Nepoch-v[0] for v in TT_dict.values()])
accs = np.array([v[Nepoch-1] if len(v)>=Nepoch else v[-1] for v in ACC_dict.values()])

idx_sort = np.argsort(dimensions)

Ds = dimensions[idx_sort]
TTs = training_times[idx_sort]
ACCs = accs[idx_sort]
lrt = LinearRegression()
lrt.fit(np.log(Ds).reshape(-1, 1),np.log(TTs).reshape(-1, 1))
para1 = np.exp(lrt.intercept_[0])
para2 = lrt.coef_[0][0]
lacc, = ax_acc.plot(Ds, ACCs+0.004, c=cacc, linewidth=1, 
                    marker='d', markersize=2.5, linestyle='--')

yticks = np.linspace(0.985,0.995,5)
yticklabels = [('%.1f%%' % (yt*100)) for yt in yticks]
ax_acc.set_yticks(yticks)
ax_acc.set_yticklabels(yticklabels)
ax_acc.set_ylim([np.min(yticks),np.max(yticks)])
ax_acc.set_ylabel('MAR')

ltt, = axt.plot(Ds, TTs, marker='o', markersize=5, c=ctt)
# axt.plot(Ds, para1*(Ds**para2), c='r', linestyle='--', linewidth=0.5)
if not change_label:
    axt.set_xlabel('dimension')
    axt.set_ylabel('training time of %d epochs (sec)' % Nepoch)
else:
    axt.set_xlabel('$D$')
    axt.set_ylabel('迭代%d次所需的训练时间 (s)' % Nepoch,fontname='Simhei')

axt.legend([ltt,lacc],['训练时间','准确率'],prop={'family':'Simhei'})
axt.set_xlim([0,np.max(Ds)+np.min(Ds)])
axt.set_position((0.1,0.1,0.75,0.88))

# %%
if change_label:
    mysavefig(figt,titlet)
