import numpy as np
import matplotlib.pyplot as plt
#from CCT_10m39b.utils.train_weights import train_weights_2


# %%
def softmax(z):
    ez = np.exp(z)
    f = np.logical_and(np.isinf(ez),ez>0)
    if np.any(f):
        ez = np.zeros(z.shape)
        ez[f] = 1
        ez[~f] = 0
    out = ez/np.sum(ez)
    return out


def cal_entropy_item(p):
    if p == 0:
        s = 0
    else:
        s = -p*np.log(p)
    return s


def cal_entropy(p_distribute):
    s = np.sum([cal_entropy_item(p) for p in p_distribute])
    return s


def cal_entropy_mat(mat_pdf):
    s = np.sum(-mat_pdf * np.log(mat_pdf),axis=1)
    return s


def root_mean_square(dx):
    rms = np.sqrt(np.mean(np.square(dx)))
    return rms


def compare_real_predict(Xtrain,Ytrain,Xtest,Ytest,model,n=0,lw=0.5):
    model.fit(Xtrain, Ytrain)
    Yp = model.predict(Xtest)
    idx_arange = np.argsort(Ytest)
    figt = plt.figure(('test_%d' % n))
    figt.clf()
    axt = figt.gca()
    l1, = axt.plot(Ytest[idx_arange],lw=lw)
    l2, = axt.plot(Yp[idx_arange],lw=lw)
    axt.legend([l1,l2],['real','predict'])
    return model,Yp


#%%
def formulate_LF_Matrix(list_LFObj,return_dict_form=False):
    dict_m = {}
    count = 0
    for lfobj in list_LFObj:
        dict_lf_e_t = lfobj.dict_lf_expanded
        keys_m = set(dict_m.keys())
        keys_t = set(dict_lf_e_t.keys())
        keys_diff = list(keys_m.difference(keys_t))
        keys_intersection = list(keys_m.intersection(keys_t))
        keys_diff_r = list(keys_t.difference(keys_m))
        keys_diff.sort()
        keys_intersection.sort()
        keys_diff_r.sort()
        for k in keys_diff:
            dict_m[k] = np.hstack([dict_m[k],float('nan')])
        for k in keys_intersection:
            dict_m[k] = np.hstack([dict_m[k],dict_lf_e_t[k]])
        for k in keys_diff_r:
            pv = np.array([float('nan')]*count)
            dict_m[k] = np.hstack([pv,dict_lf_e_t[k]])
        count += 1

    if return_dict_form:
        return dict_m
    else:
        LFkeys = list(dict_m.keys())
        LFM = np.vstack(list(dict_m.values())).T
        return LFkeys,LFM

# %%
vec_conf = np.linspace(0.01,0.99,100)
def pick_Ys_byPmat(Pmat, Y_train, vconf=vec_conf):
    N_test = Pmat.shape[0]
    Lc = vconf.size
    Ys = np.zeros([Lc, N_test])
    count = 0
    idx_sort_t = np.argsort(Y_train)
    yc = Y_train[idx_sort_t]
    Pmat_sorted = Pmat[:,idx_sort_t]
    PDFmat_sorted = np.cumsum(Pmat_sorted,axis=1)
    for ct in vconf:
        ys_t = np.zeros(N_test)
        count_t = 0
        for i in range(N_test):
            pdfc = PDFmat_sorted[i,:]
            idxt = np.where(pdfc<(1-ct))[0]
            if len(idxt)==0:
                ys_t[count_t] = yc[0]
            else:
                ys_t[count_t] = yc[np.max(idxt)]
            count_t += 1
        Ys[count, :] = ys_t
        count += 1
    return Ys

