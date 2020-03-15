import numpy as np
import matplotlib.pyplot as plt
from MDKernelEstimator.Estimator import Estimator, DistanceTrainer
from CCT_10m39b.DataAnalysis.ExtractData import X_train, Y_train, X_test, Y_test
from CCT_10m39b.utils.tools import root_mean_square
import os
import scipy.io as scio


# %%
# dist_trainer = DistanceTrainer(epochs=1000, batch_size=1000, reg_alpha=0)  # 4.381
dist_trainer = DistanceTrainer(epochs=1000, batch_size=2000, reg_alpha=0)  # 4.086
# dist_trainer = DistanceTrainer(epochs=10000, batch_size=500, reg_alpha=0)  # 3.700
dist_trainer.fit(X_train, Y_train)


# %%
mat_ml = os.path.join('..','RESULTS','CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate','ml.mat')
scio.savemat(mat_ml,{'KN':dist_trainer.ph.r_KN,'m':dist_trainer.ph.r_m})

# %%
estimator = Estimator(X_train, Y_train, dist_trainer.ph.r_KN, dist_trainer.ph.r_m, dist_trainer.cal_dist_mode)
Y_test_predict = estimator.predict(X_test)
idx_t = np.argsort(Y_test.flatten())
fig_t = plt.figure('compare_test')
fig_t.clf()
axt = fig_t.gca()
l_test, = axt.plot(Y_test.flatten()[idx_t])
l_test_predict, = axt.plot(Y_test_predict.flatten()[idx_t])
axt.legend([l_test, l_test_predict], ['test', 'predict'])
fig_t.show()

error_test = root_mean_square(Y_test - Y_test_predict)

print(error_test)

# %%
from sklearn.linear_model import RidgeCV

reg = RidgeCV()                 # 6.57
# reg = LassoCV()                 # 7.38
# reg = DecisionTreeRegressor()   # 8.96
# reg = MLPRegressor()            # 17.9
reg.fit(X_train, Y_train)
Y_test_predict_ridge = reg.predict(X_test)
error_test_comp = root_mean_square(Y_test - Y_test_predict_ridge)

print(error_test_comp)
