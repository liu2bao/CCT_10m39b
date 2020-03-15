import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as col
import matplotlib.cm as cm
import numpy as np
from scipy.interpolate import interp1d
import os
from CCT_10m39b.Simulation.gadgets import get_path_file_figures

path_figures_thesis = get_path_file_figures()
postfix_save = '.pdf'
mapping_color = {
    'Proposed': '#1f77b4',
    'ANN': '#ff7f0e',
    'LASSO': '#2ca02c',
    'BR': '#d62728',
    'CART': '#9467bd'
}

def hex2int(h):
    return int(h.replace('#', ''), 16)


def int2hex(i):
    return str(hex(i)).replace('0x', '#')

def mysavefig(fig,figname=None,figpath=path_figures_thesis,postfix=postfix_save):
    if figname is None:
        figname = fig.get_label()
    figpatht = os.path.join(figpath,figname+postfix)
    fig.savefig(figpatht)


pos_all = [0.12,0.12,0.85,0.85]
# bluecolor = '#BBFFFF'
# redcolor = '#FF6A6A'
bluecolor = 'blue'
redcolor = 'red'
purplecolor = 'purple'
whitecolor = 'white'
my_cmap = col.LinearSegmentedColormap.from_list('my_cmap', [bluecolor, redcolor])
my_cmap_red = col.LinearSegmentedColormap.from_list('my_cmap_red', [redcolor, whitecolor])
my_cmap_blue = col.LinearSegmentedColormap.from_list('my_cmap_blue', [bluecolor, whitecolor])
my_cmap_blue_flip = col.LinearSegmentedColormap.from_list('my_cmap_blue_flip', [whitecolor, bluecolor])

my_font = {'size': 10, 'family':'Times New Roman'}

h1 = 0.499
h2 = 0.501
cdict = {'blue': ((0.0, 1.0, 1.0),
                  (h1, 0.0, 0.0),
                  (h2, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),

         'green': ((0.0, 1.0, 1.0),
                   (h1, 0.0, 0.0),
                   (h2, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'red': ((0.0, 1.0, 1.0),
                 (h1, 1.0, 1.0),
                 (h2, 0.0, 0.0),
                 (1.0, 1.0, 1.0))
         }

cdict2 = {'blue': ((0.0, 0.0, 0.0),
                   (h1, 0.0, 0.0),
                   (h2, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (h1, 0.0, 0.0),
                    (h2, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'red': ((0.0, 1.0, 1.0),
                  (h1, 1.0, 1.0),
                  (h2, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),

          'alpha': ((0.0, 0.0, 0.0),
                    (h1, 1.0, 1.0),
                    (h2, 1.0, 1.0),
                    (1.0, 0.0, 0.0))
          }

pm = 0.2
am = 0.8
cdict3 = {'blue': ((0.0, 0.0, 0.0),
                   (pm, am, am),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (pm, am, am),
                    (0.5, 1.0, 1.0),
                    (1 - pm, am, am),
                    (1.0, 0.0, 0.0)),

          'red': ((0.0, 1.0, 1.0),
                  (0.5, 1.0, 1.0),
                  (1 - pm, am, am),
                  (1.0, 0.0, 0.0)),

          }

my_cmap_all = col.LinearSegmentedColormap('my_cmap_all', cdict3)
# my_cmap_all = cm.coolwarm

# extra arguments are N=256, gamma=1.0
cm.register_cmap(cmap=my_cmap)
cm.register_cmap(cmap=my_cmap_red)
cm.register_cmap(cmap=my_cmap_blue)

def plot_fore(CCTs,Ytest_new,Y_predict,vec_confidence,change_label=False):
    if change_label:
        title_t = 'PDF-colored'
    else:
        title_t = 'fore'
    Ntest = Ytest_new.size
    fig_fore = plt.figure(title_t)
    fig_fore.clf()
    ax_fore = fig_fore.gca()
    l = []
    le = []
    # idx_sort = np.argsort(Ytest_new)
    idx_sort = np.argsort(np.median(CCTs, axis=0))
    ccts_ori = np.zeros(Ntest)
    x_fill_di = np.arange(0, Ntest)
    x_fill = np.hstack([x_fill_di, x_fill_di[-1::-1]])

    count = 0
    for ct in vec_confidence:
        ccts_t = CCTs[count, :]
        if count > 0:
            # ccts_fill = np.hstack([ccts_ori[idx_sort],ccts_ori[idx_sort[-1::-1]]])
            colt = my_cmap_all(ct)
            ax_fore.fill_between(x_fill_di, ccts_ori[idx_sort], ccts_t[idx_sort],
                                 facecolor=colt, edgecolor='none')
        # lt, = ax_fore.plot(ccts_t[idx_sort],lw=0.3)
        # l.append(lt)
        le.append('%.3f' % ct)
        ccts_ori = ccts_t
        count += 1

    '''
    l_conf_show = []
    le_conf_show = []
    conf_show = np.array([0,0.25,0.75,1])
    for conft in conf_show:
        pt = 1-conft
        idx_pt = int((Lc-1)*conft)
        lt, = ax_fore.plot(CCTs[idx_pt,:][idx_sort],c=my_cmap_all(conft))
        l_conf_show.append(lt)
        le_t = '$\epsilon$ = %d%%' % (conft*100)
        le_conf_show.append(le_t)
    
    ax_fore.legend(l_conf_show,le_conf_show)
    '''

    CCT_median = np.median(CCTs, axis=0)
    CCT_mean = np.mean(CCTs, axis=0)
    #CCT_mean = Y_predict
    # ax_fore.legend(l,le)
    l_real, = ax_fore.plot(Ytest_new[idx_sort], lw=1.5, alpha=0.6, c='k')
    l_median, = ax_fore.plot(CCT_median[idx_sort], lw=2, c='#00FF00', linestyle='--')
    l_mean, = ax_fore.plot(CCT_mean[idx_sort], lw=2, c='#FFFF00', linestyle='-.')
    if change_label:
        legends = [r'$\overline{Y}_j$',
                   r'$Y_{j}^{\rm test}$',
                   r'$\hat{Y}_{j}^{\rm test}$']
    else:
        legends = [r'$\overline{y}_{{\rm{CCT}},m}$',
                   r'$\hat{y}_{{\rm{CCT}},m}^{\rm{con}}(50\%)$',
                   r'$\hat{y}_{{\rm{CCT}},m}$']
    ax_fore.legend([l_real, l_median, l_mean],legends,prop=my_font,loc='upper left')
    ax_fore.set_xticks([])

    sm = plt.cm.ScalarMappable(cmap=my_cmap_all, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    ax_fore.set_position([0.12, 0.2, 0.84, 0.77])
    cax = fig_fore.add_axes([0.12, 0.1, 0.84, 0.03])
    cbar = fig_fore.colorbar(sm, ax=ax_fore, cax=cax, orientation='horizontal')
    xticks_cax = np.linspace(0, 1, 5)
    xticklabels_cax = [('%d%%' % (t * 100)) for t in xticks_cax]
    cbar.set_ticks(xticks_cax)
    cbar.set_ticklabels(xticklabels_cax)
    if change_label:
        cax.set_xlabel('$\sigma$', fontdict=my_font)
        ax_fore.set_ylabel(r'CCT  (s)', fontdict=my_font)
        ax_fore.set_xlabel('$j$')
    else:
        cax.set_xlabel('$\epsilon$', fontdict=my_font)
        ax_fore.set_ylabel(r'$y_{{\rm{CCT}},m}$ (s)', fontdict=my_font)
        ax_fore.set_xlabel('$m$')

    minY = np.percentile(Ytest_new, 5)
    maxY = np.percentile(Ytest_new, 95)
    idxt = np.where(np.logical_and(Ytest_new[idx_sort] > minY,
                                   Ytest_new[idx_sort] < maxY))[0]
    ax_fore.set_xlim([np.min(idxt), np.max(idxt)])
    ax_fore.set_ylim([minY - 0.01, maxY + 0.01])
    if change_label:
        # figpatht = os.path.join(path_figures_thesis,title_t+postfix_save)
        mysavefig(fig_fore,title_t,postfix='.png')


def plot_pdf_3d(mat_pdf_s, vec_y_pdf, ylimt=(0.2,0.38)):
    titlet = 'PDF-3D'
    idx_sel = np.logical_and(vec_y_pdf>=ylimt[0],vec_y_pdf<=ylimt[1])
    vec_y_pdf_r = vec_y_pdf[idx_sel]
    mat_pdf_r = mat_pdf_s[:,idx_sel]
    Ntest,Ndiv = mat_pdf_r.shape
    vec_j = np.arange(Ntest)
    figt = plt.figure(titlet)
    axt = figt.add_subplot(111,projection='3d')
    for j in vec_j:
        axt.plot(j * np.ones(Ndiv), vec_y_pdf_r, mat_pdf_r[j, :])
    axt.set_ylim([0.2,0.38])


def mesh_cdf_3d(mat_cdf_s, vec_y_pdf, ylimt=(0.2, 0.4),
                cmap=my_cmap_all,elev=None,azim=None,pdf=False,save=False,
                exXY=False):
    if pdf:
        titlet = 'PDF-3D-MESH'
    else:
        titlet = 'CDF-3D-MESH'
    if cmap is None:
        cmap = cm.coolwarm
    idx_sel = np.logical_and(vec_y_pdf>=ylimt[0],vec_y_pdf<=ylimt[1])
    vec_y_pdf_r = vec_y_pdf[idx_sel]
    mat_pdf_r = mat_cdf_s[:, idx_sel]
    Ntest,Ndiv = mat_pdf_r.shape
    vec_j = np.arange(Ntest)
    XX,YY = np.meshgrid(vec_j, vec_y_pdf_r)
    XX = XX.T
    YY = YY.T
    '''
    if not pdf:
        idx_sel_t = np.logical_and(mat_pdf_r>0.01,mat_pdf_r<0.99)
        mat_pdf_r[~idx_sel_t] = np.nan
    '''
    figt = plt.figure(titlet)
    figt.clf()
    axt = figt.add_subplot(111,projection='3d')
    axt.view_init(elev=elev,azim=azim)
    xticks = [0,Ntest-1]
    xticklabels = ['1',r'$N^{\rm test}$']
    yticks = np.linspace(ylimt[0],ylimt[1],5)
    yticklabels = [('%.2f' % yt) for yt in yticks]
    if exXY:
        axt.plot_surface(YY,XX,mat_pdf_r,cmap=cmap)
        axt.set_xlim(ylimt)
        axt.set_xticks(yticks)
        axt.set_xticklabels(yticklabels)
        axt.set_yticks(xticks)
        axt.set_yticklabels(xticklabels)
        axt.set_ylabel('$j$')
        axt.set_xlabel('$y$ (s)')
    else:
        axt.plot_surface(XX,YY,mat_pdf_r,cmap=cmap)
        axt.set_ylim(ylimt)
        axt.set_yticks(yticks)
        axt.set_yticklabels(yticklabels)
        axt.set_xticks(xticks)
        axt.set_xticklabels(xticklabels)
        axt.set_xlabel('$j$')
        axt.set_ylabel('$y$ (s)')
    zticks = np.linspace(0,1,6)
    zticklabels = [('%d%%' % np.round(zt*100)) for zt in zticks]
    if not pdf:
        axt.set_zticks(zticks)
        axt.set_zticklabels(zticklabels)
        axt.set_zlabel(r'$\sigma=1-{\rm CDF}$')
    else:
        axt.set_zlabel('PDF ($s^{-1}$)')
    # axt.contourf(XX.T,YY.T,mat_pdf_r,cmap=cmap,zdir='Z',offset=-0.2)
    # axt.set_xticks([])
    if not pdf:
        pos_t = [0.07,0.07,0.9,0.85]
        axtitlet = r'$\sigma(X_j^{\rm test}, y) = 1-{\rm CDF}(X_j^{\rm test}, y)$'
    else:
        pos_t = [0.12,0.07,0.85,0.85]
        axtitlet = r'${\rm PDF}(X_j^{\rm test}, y)$'
    axt.set_title(axtitlet)
    axt.set_position(pos_t)
    if save:
        mysavefig(figt,titlet)
    return axt




def plot_acc_vs_con(vec_confidence,errs_conf,conservs_conf,change_label=False):
    if change_label:
        title_t = 'ACC-VS-CONS'
    else:
        title_t = 'acc vs cons'
    startConf = 0.5
    endConf = 0.8
    idx_before = np.where(vec_confidence < startConf)
    idx_after = np.where(vec_confidence > endConf)
    idx_within = np.where(np.logical_and(vec_confidence >= startConf, vec_confidence <= endConf))

    lw_dash = 1
    alpha_dash = 0.3
    alpha_out = 0.3
    lw_ax = 1
    lw = 3.5
    lw_emph = 3.5
    p = 0.15
    P = [p, p, 1 - 2 * p, 1 - 2 * p]
    PL = [0, -4.5 * p, 1, 1 - 2 * p]
    # color_acc = '#98FB98'
    # color_con = '#FF69B4'
    color_acc = 'blue'
    color_con = 'red'
    fig_acc_cons = plt.figure(title_t)
    fig_acc_cons.clf()
    ax_acc = fig_acc_cons.gca()
    ax_con = plt.twinx(ax_acc)
    ax_acc.spines['left'].set_color(color_acc)
    ax_con.spines['right'].set_color(color_con)
    ax_acc.spines['right'].set_visible(False)
    ax_con.spines['left'].set_visible(False)
    ax_acc.set_position(P)
    ax_con.set_position(P)
    ax_acc.spines['left'].set_linewidth(lw_ax)
    ax_con.spines['right'].set_linewidth(lw_ax)
    ax_acc.spines['bottom'].set_linewidth(lw_ax)
    ax_acc.spines['top'].set_visible(False)
    ax_con.spines['top'].set_visible(False)

    ax_acc.plot(vec_confidence[idx_before], errs_conf[idx_before], c=color_acc, lw=lw, alpha=alpha_out)
    ax_con.plot(vec_confidence[idx_before], conservs_conf[idx_before], c=color_con, lw=lw, alpha=alpha_out)
    ax_acc.plot(vec_confidence[idx_after], errs_conf[idx_after], c=color_acc, lw=lw, alpha=alpha_out)
    ax_con.plot(vec_confidence[idx_after], conservs_conf[idx_after], c=color_con, lw=lw, alpha=alpha_out)
    l_acc, = ax_acc.plot(vec_confidence[idx_within], errs_conf[idx_within], c=color_acc, lw=lw_emph)
    l_con, = ax_con.plot(vec_confidence[idx_within], conservs_conf[idx_within], c=color_con, lw=lw_emph)

    if change_label:
        PLt = r'$P_{\rm L}$'
        xsymt = '$\sigma$'
    else:
        PLt = '$P_L$'
        xsymt = '$\epsilon$'
    lgt = ['RMSE', PLt]

    ax_acc.set_position([0.12, 0.1, 0.76, 0.88])
    ax_acc.legend([l_acc, l_con], lgt, loc='lower center',
                  bbox_to_anchor=(0.2, 0.7, 0.1, 0.2))

    xlim = [0.05, 0.95]
    err_t = errs_conf[np.logical_and(vec_confidence >= xlim[0], vec_confidence <= xlim[1])]
    ylim_err = [np.min(err_t) - 0.001, np.max(err_t) + 0.001]
    ax_acc.set_xlim(xlim)
    xtickst = np.linspace(xlim[0], xlim[1], 7)
    xtickst = np.hstack([xtickst, [startConf, endConf]])
    xtickst = np.unique(np.round(xtickst, 2))
    xticklabelst = [('%d%%' % round(t * 100)) for t in xtickst]
    ax_acc.set_xticks(xtickst)
    ax_acc.set_xticklabels(xticklabelst)
    ax_acc.set_ylim(ylim_err)
    ax_acc.set_xlabel(xsymt)
    ax_acc.set_ylabel('RMSE (s)')
    ax_con.set_ylabel(PLt)

    ytickst_acc = np.linspace(0, 0.025, 6)
    yticklabelst_acc = [('%.3f' % t) for t in ytickst_acc]
    ax_acc.set_yticks(ytickst_acc)
    ax_acc.set_yticklabels(yticklabelst_acc)
    ax_acc.set_ylim([ytickst_acc[0], ytickst_acc[-1]])
    for conft in [startConf, endConf]:
        ax_acc.plot([conft, conft], [ytickst_acc[0], ytickst_acc[-1]],
                    c='k', linestyle='--', alpha=alpha_dash, lw=lw_dash)

    axs = [ax_acc, ax_con]
    colors = [color_acc, color_con]
    countt = 0
    for idxt in [np.min(idx_within), np.max(idx_within)]:
        xt = vec_confidence[idxt]
        errt = errs_conf[idxt]
        cont = conservs_conf[idxt]
        xss = [[xlim[0], xt], [xt, xlim[1]]]
        ys = [errt, cont]
        for hh in range(len(axs)):
            axt = axs[hh]
            axt.plot(xss[hh], [ys[hh], ys[hh]], lw=lw_dash, c=colors[hh], linestyle='-.', alpha=alpha_dash)
            axt.scatter(xt, ys[hh], c=colors[hh], s=50)
        countt += 1

    ytickst_con = np.linspace(0, 1, 5)
    yticklabelst_con = [('%d%%' % (t * 100)) for t in ytickst_con]
    ax_con.set_yticks(ytickst_con)
    ax_con.set_yticklabels(yticklabelst_con)
    ax_con.set_ylim([0 - 0.01, 1 + 0.01])

    '''
    s = 100
    ax_acc.scatter(vec_confidence,errs_conf,c=np.array([my_cmap_all(t) for t in vec_confidence]),s=s)
    ax_con.scatter(vec_confidence,conservs_conf,c=np.array([my_cmap_all(t) for t in vec_confidence]),s=s)
    cax = fig_acc_cons.add_axes([0.13, 0.06, 0.75, 0.03])
    cbar = fig_acc_cons.colorbar(sm, ax=ax_acc, cax=cax,orientation='horizontal')
    xticks_cax = np.linspace(0,1,5)
    xticklabels_cax = [('%d%%' % (t*100)) for t in xticks_cax]
    cbar.set_ticks(xticks_cax)
    cbar.set_ticklabels(xticklabels_cax)
    #cax.set_xlabel('$\epsilon$')
    '''

    if change_label:
        mysavefig(fig_acc_cons, title_t)

    return fig_acc_cons


def plot_err_multi_model(ETT,Npicks,N):
    Ncolfig = 2
    for k, ETt in ETT.items():
        fig_bp = plt.figure('errors distribution : %s' % k)
        fig_bp.clf()
        # ax_bp = fig_bp.gca()
        dx_bp = 0.05
        w_bp = 0.03
        count = 0
        Nm = len(ETt)
        Np = len(Npicks)
        xticklabels = ['%d%%' % (t / N * 100 + 1) for t in Npicks]
        Nfig = np.ceil(Nm / Ncolfig)
        for kk, vv in ETt.items():
            ylim_0 = np.percentile(vv.flatten(), 0.5)
            ylim_1 = np.percentile(vv.flatten(), 99.5)
            ax_bp_t = fig_bp.add_subplot(Nfig, 2, count + 1)
            ax_bp_t.boxplot(vv.T)
            ax_bp_t.plot(np.arange(1, Np + 1), np.mean(np.abs(vv), axis=1), lw=2, linestyle='--', c='r', marker='o')
            # ax_bp.boxplot(v.T,positions=np.arange(1,Np+1)+(count-(Nm-1)/2)*dx_bp,widths=w_bp)
            count += 1
            ax_bp_t.set_xticks(np.arange(1, Np + 1))
            ax_bp_t.set_xticklabels(xticklabels)
            ax_bp_t.set_title(kk)
            ax_bp_t.set_ylim([ylim_0, ylim_1])

        fig_bp.subplots_adjust(wspace=0.2, hspace=0.2)


def plot_ETEST_mean(ETEST_mean,Npicks,N,change_label=False):
    if change_label:
        titlet = 'ETEST-MEAN'
    else:
        titlet = 'ETEST_mean'
    fig_ETEST_mean = plt.figure(titlet)
    fig_ETEST_mean.clf()
    ax_t = fig_ETEST_mean.add_subplot(111)
    lines = []
    legends = []
    props = np.array(Npicks) / N
    for k, v in ETEST_mean.items():
        if k == 'Proposed':
            lwt = 4
        else:
            lwt = 2
        markersize = lwt*2
        lt, = ax_t.plot(props, v*1000, lw=lwt, marker='o',
                        markersize=markersize,c=mapping_color[k])
        lines.append(lt)
        legends.append(k)
        
    if change_label:
        legends[legends.index('Proposed')] = 'NWKR'
        legends[legends.index('BR')] = 'BLR'
        xlabel_t = r'$p^{\rm train}$'
    else:
        xlabel_t = r'${\alpha}_{\rm{training}}$'
    Ntick = 5
    xticks = np.linspace(props[0], props[-1], Ntick)
    xticklabels = ['%d%%' % (t * 100 + 1) for t in xticks]
    ax_t.set_xticks(xticks)
    ax_t.set_xticklabels(xticklabels)
    ax_t.legend(lines, legends, ncol=int(len(legends) / 2), loc='lower center')
    ax_t.set_ylim([0, 16.5])
    ax_t.set_xlabel(xlabel_t)
    ax_t.set_ylabel('RMSE (ms)')
    ax_t.set_position(pos_all)
    if change_label:
        mysavefig(fig_ETEST_mean, titlet)


def plot_ETEST_mean_m(ETEST_mean, props):
    fig_ETEST_mean = plt.figure('ETEST_mean')
    fig_ETEST_mean.clf()
    ax_t = fig_ETEST_mean.add_subplot(111)
    lines = []
    legends = []
    for k, v in ETEST_mean.items():
        if k == 'Proposed':
            lwt = 2
        else:
            lwt = 2
        if v[0][0]>props[0]:
            v[0] = np.hstack([props[0],v[0]])
            v[1] = np.hstack([v[1][0],v[1]])
        if v[0][-1]<props[-1]:
            v[0] = np.hstack([v[0],props[-1]])
            v[1] = np.hstack([v[1],v[1][-1]])
        func_interp = interp1d(v[0],v[1])
        y_t = func_interp(props)
        lt, = ax_t.plot(props,y_t, lw=lwt, marker='o',c=mapping_color[k])
        lines.append(lt)
        legends.append(k)

    Ntick = 5
    xticks = np.linspace(props[0], props[-1], Ntick)
    xticklabels = ['%d%%' % (t * 100) for t in xticks]
    ax_t.set_xticks(xticks)
    ax_t.set_xticklabels(xticklabels)
    ax_t.legend(lines, legends, ncol=int(len(legends) / 2), loc='lower center')
    ax_t.set_ylim([0, 0.0165])
    ax_t.set_xlabel(r'${\alpha}_{\rm{training}}$')
    ax_t.set_ylabel('RMSE')
    ax_t.set_position(pos_all)


def plot_ETEST_err(ETEST_new,dens_err):
    lines = []
    legends = []
    fig_err = plt.figure('err')
    fig_err.clf()
    ax_err = fig_err.add_subplot(111)
    for k, v in ETEST_new.items():
        vs = np.sort(v[-1, :])
        lt, = ax_err.plot(vs, dens_err, c=mapping_color[k])
        lines.append(lt)
        legends.append(k)

    ax_err.plot([0, 0], [0, 1], linestyle='--', c='k', alpha=0.5)

    Ntick = 5
    yticks = np.linspace(0, 1, Ntick)
    yticklabels = ['%d%%' % (t * 100) for t in yticks]
    ax_err.set_yticks(yticks)
    ax_err.set_yticklabels(yticklabels)

    ax_err.legend(lines, legends, loc='lower right')
    ax_err.set_xlim([-0.02, 0.02])
    ax_err.set_ylim([0, 1])
    ax_err.set_xlabel('$E_m$')
    ax_err.set_ylabel('CDF')
    ax_err.set_position(pos_all)


def plot_err_hist(methods,ETMe,Nbins,close=True):
    lines = []
    legends = []
    fig_err_hist = plt.figure('err_hist')
    fig_err_hist.clf()
    ax_err_hist = fig_err_hist.add_subplot(111)
    ns, binss, patches = ax_err_hist.hist(ETMe.T, bins=Nbins)
    fig_err_hist.legend(methods)
    if close:
        plt.close(fig_err_hist)
    return ns,binss,patches


def plot_err_hist_curve(methods, idx_proposed, freqs, binss, Xbinss,change_label=False):
    if change_label:
        titlet = 'ERR-HIST-CURVE'
    else:
        titlet = 'err_hist_curve'
    fig_err_hist_curve = plt.figure(titlet)
    fig_err_hist_curve.clf()
    ax_err_hist_curve = fig_err_hist_curve.add_subplot(111)
    for hh in range(len(methods)):
        if hh == idx_proposed:
            lwt = 2
        else:
            lwt = 1
        fdt = freqs[hh, :] / np.mean(np.diff(binss))
        ax_err_hist_curve.plot(Xbinss, fdt, lw=lwt, marker='s', markersize=lwt * 2,
                               c=mapping_color[methods[hh]])

    # ax_err_hist_curve.plot(Xbinss,freqs.T,lw=1)
    ax_err_hist_curve.set_position(pos_all)
    if change_label:
        methods_new = methods.copy()
        methods_new[methods.index('Proposed')] = 'NWKR'
        methods_new[methods.index('BR')] = 'BLR'
    else:
        methods_new = methods.copy()
    fig_err_hist_curve.legend(methods_new, loc='upper right',
                              bbox_to_anchor=(0.75, 0.6, 0.2, 0.3))
    ax_err_hist_curve.set_xlim([-25, 25])
    ylimt = ax_err_hist_curve.get_ylim()
    ax_err_hist_curve.plot([0, 0], ylimt, c='k', linestyle='--', alpha=0.5)
    ax_err_hist_curve.set_ylim(ylimt)
    if change_label:
        ax_err_hist_curve.set_xlabel(r'${\rm{E}}_j$ (ms)')
        ax_err_hist_curve.set_ylabel(r'频率密度 (${\rm{ms}}^{-1}$)',fontname='Simhei')
        ax_err_hist_curve.set_xlim([-20, 20])
        mysavefig(fig_err_hist_curve, titlet)
    else:
        ax_err_hist_curve.set_xlabel(r'${\rm{E}}_m$ (ms)')
        ax_err_hist_curve.set_ylabel(r'Frequency density (${\rm{ms}}^{-1}$)')


def plot_err_hist_curve_m(data):
    fig_err_hist_curve = plt.figure('err_hist_curve')
    fig_err_hist_curve.clf()
    ax_err_hist_curve = fig_err_hist_curve.add_subplot(111)
    methods = list(data.keys())
    for m,d in data.items():
        if m=='Proposed':
            lwt = 2
        else:
            lwt = 1
        ax_err_hist_curve.plot(d[0], d[1], lw=lwt, marker='s', markersize=lwt * 2, c=mapping_color[m])

    # ax_err_hist_curve.plot(Xbinss,freqs.T,lw=1)
    ax_err_hist_curve.set_position(pos_all)
    fig_err_hist_curve.legend(methods, loc='upper right', bbox_to_anchor=(0.75, 0.6, 0.2, 0.3))
    ax_err_hist_curve.set_xlim([-20, 20])
    ylimt = ax_err_hist_curve.get_ylim()
    ax_err_hist_curve.plot([0, 0], ylimt, c='k', linestyle='--', alpha=0.5)
    ax_err_hist_curve.set_ylim(ylimt)
    ax_err_hist_curve.set_xlabel(r'${\rm{E}}_m$ (ms)')
    ax_err_hist_curve.set_ylabel(r'Frequency density (${\rm{ms}}^{-1}$)')


def plot_err_hist_curve_3d(methods,IDX_a,XbinssM,freqs,binss):
    fig_err_hist_curve_3d = plt.figure('err_hist_curve_3d')
    fig_err_hist_curve_3d.clf()
    ax_err_hist_curve_3d = fig_err_hist_curve_3d.add_subplot(111, projection='3d')
    for hh in range(len(methods)):
        idx_t = IDX_a[hh, :]
        Xb_t = XbinssM[hh, :]
        freq_t = freqs[hh, :]
        freq_dens_t = freq_t / np.mean(np.diff(binss))
        ax_err_hist_curve_3d.plot(idx_t, Xb_t, freq_dens_t, lw=2, marker='d', markersize=2, alpha=0.5)
    # ax_err_hist_curve.plot(Xbinss,freqs.T,lw=1)
    fig_err_hist_curve_3d.legend(methods)


def plot_acc(ACCTEST_new,dens_err):
    lines = []
    legends = []
    fig_acc = plt.figure('acc')
    fig_acc.clf()
    ax_acc = fig_acc.add_subplot(111)
    for k, v in ACCTEST_new.items():
        vs = np.sort(v)
        lt, = ax_acc.plot(vs, dens_err, c=mapping_color[k])
        lines.append(lt)
        legends.append(k)
    Ntick = 5
    Ntickx = 4
    yticks = np.linspace(0, 1, Ntick)
    yticklabels = ['%d%%' % (t * 100) for t in yticks]
    xticks = np.linspace(0.94, 1, Ntickx)
    xticklabels = ['%d%%' % (t * 100) for t in xticks]
    ax_acc.set_yticks(yticks)
    ax_acc.set_yticklabels(yticklabels)
    ax_acc.set_xticks(xticks)
    ax_acc.set_xticklabels(xticklabels)

    ax_acc.legend(lines, legends, loc='upper left')
    ax_acc.set_xlim([xticks[0], xticks[-1]])
    ax_acc.set_ylim([0, 1])
    ax_acc.set_xlabel('$e_m$')
    ax_acc.set_ylabel('CDF')
    ax_acc.set_position(pos_all)


def plot_err_rate(ERRRATETEST_new,dens_err):
    lines = []
    legends = []
    fig_err_rate = plt.figure('err_rate')
    fig_err_rate.clf()
    ax_err_rate = fig_err_rate.add_subplot(111)
    for k, v in ERRRATETEST_new.items():
        vs = np.sort(v)
        lt, = ax_err_rate.plot(vs, dens_err, c=mapping_color[k])
        lines.append(lt)
        legends.append(k)

    ax_err_rate.plot([0, 0], [0, 1], linestyle='--', c='k', alpha=0.5)

    Ntick = 5
    yticks = np.linspace(0, 1, Ntick)
    yticklabels = ['%d%%' % (t * 100) for t in yticks]
    Ntickx = 5
    xticks = np.linspace(-0.08, 0.08, Ntickx)
    xticklabels = ['%d%%' % (t * 100) for t in xticks]
    ax_err_rate.set_yticks(yticks)
    ax_err_rate.set_yticklabels(yticklabels)
    ax_err_rate.set_xticks(xticks)
    ax_err_rate.set_xticklabels(xticklabels)

    ax_err_rate.legend(lines, legends, loc='lower right')

    ax_err_rate.set_xlim([xticks[0], xticks[-1]])
    ax_err_rate.set_ylim([0, 1])
    ax_err_rate.set_xlabel('$e_m$')
    ax_err_rate.set_ylabel('CDF')
    ax_err_rate.set_position(pos_all)


