import sys
sys.path.append('F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')
sys.path.append('F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')
from CCT_10m39b.Simulation.plotfuncs import mysavefig
from CCT_10m39b.Plot10m39b.plot10m39b import plot_10m39b_basic
from CCT_10m39b.Plot10m39b.plotSMIB import plot_SMIB_basic
import matplotlib.pyplot as plt


if __name__=='__main__':
    postfixt='.eps'
    fig_10m39b = plt.figure('10M39B',figsize=(8,8))
    fig_smib = plt.figure('SMIB',figsize=(5.4,2))
    plt.close(fig_10m39b)
    plt.close(fig_smib)
    plot_10m39b_basic(fig_10m39b)
    plot_SMIB_basic(fig_smib)
    mysavefig(fig_10m39b,'SLG-IEEE10M39B',postfix=postfixt)
    mysavefig(fig_smib, 'SLG-SMIB', postfix=postfixt)

