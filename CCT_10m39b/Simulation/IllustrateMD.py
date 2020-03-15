import sys
sys.path.append('F:\Programs\PythonWorks\PowerSystem\CCT_10m39b')
sys.path.append('F:\Programs\PythonWorks\PowerSystem\MDKernelEstimator')
import numpy as np
import matplotlib.pyplot as plt
from CCT_10m39b.Simulation.plotfuncs import mysavefig


def get_rotM(rot_t):
    rotM_t = np.array([[np.cos(rot_t),-np.sin(rot_t)],[np.sin(rot_t),np.cos(rot_t)]])
    return rotM_t


def draw_axis(ax,xylim,transM=None,alpha=1,ls='-',hw=None,hl=None,c='k'):
    XYaxis = np.array([[xylim[0,0],0],[xylim[1,0],0],
                       [0,xylim[0,1]],[0,xylim[1,1]]])
    if hw is None:
        hw = (xylim[1,0]-xylim[0,0])/40
    if hl is None:
        hl = hw*2
    if transM is not None:
        XYaxis = np.matmul(XYaxis,transM)
    for XYaxis_part in [XYaxis[0:2,:],XYaxis[2:,:]]:
        ax.arrow(XYaxis_part[0,0],XYaxis_part[0,1],
                 XYaxis_part[1,0]-XYaxis_part[0,0],
                 XYaxis_part[1,1]-XYaxis_part[0,1],
                 length_includes_head=True, linestyle=ls,
                 head_width=hw, head_length=hl, alpha=alpha,
                 color=c)
    return XYaxis


N = 1000
R = np.random.rand(N)
Theta = np.random.rand(N)*2*np.pi

X = R*np.cos(Theta)
Y = R*np.sin(Theta)
XY = np.vstack([X,Y]).T

AmpX = 0.8
AmpY = np.random.rand()+2
Rot = (np.random.rand()+1)*0.125*np.pi
AmpM = np.diag([AmpX,AmpY])
RotM = get_rotM(Rot)
TransM = np.matmul(AmpM,RotM)

XY_new = np.matmul(XY,TransM)
XY = np.matmul(XY_new,np.linalg.inv(TransM))
Xn = XY_new[:,0]
Yn = XY_new[:,1]

plt.close(plt.gcf())
figt = plt.figure('MD-transformation',figsize=(7,4))
figt.clf()

mar_mid = 0.1
mar_side = 0.02
mar_lower = 0.02
h_all = 1-mar_lower*2
w_ax = (1-mar_side*2-mar_mid)/2
ax_0 = figt.add_subplot(131,position=[mar_side,mar_lower,w_ax,h_all])
ax_1 = figt.add_subplot(133,position=[0.5+mar_mid/2,mar_lower,w_ax,h_all])
ax_mid = figt.add_subplot(132,position=[mar_side+w_ax,mar_lower,mar_mid,h_all])

axs = [ax_0,ax_1]
colors = ['#FFFF00','#00FFFF']
colors_axis = ['#FF0000','#000080']
XYs = [XY_new,XY]
transMs = [np.linalg.inv(TransM),TransM]
XYlims = [np.array([[-1.1,-1.5],[1.1,1.5]]),
          np.array([[-1.2,-1.2],[1.2,1.2]])]
flag_mark = [False,True]

rotNote_rad = np.pi/12
rotNote_div = 20
noteColor = '#008B00'
oA = 1.6

pick_func = lambda x: np.argmin(np.sum(np.square(XY-x),axis=1))
Npick1 = pick_func([0.7,0.05])
Npick2 = pick_func([-0.8,0.35])
Npicks = [Npick1, Npick2]


AmpMar = 1.1
count = 0
for hh in range(len(axs)):
    ax = axs[hh]
    ax_twin = axs[1-hh]
    c = colors[hh]
    XYt = XYs[hh]
    Rott = (hh-0.5)*2*Rot
    RotMt = np.array([[np.cos(Rott),-np.sin(Rott)],[np.sin(Rott),np.cos(Rott)]])
    xylimt = XYlims[hh]
    #xylimt = np.array([np.min(XYt,axis=0),np.max(XYt,axis=0)])*1.2
    
    ax.scatter(XYt[:,0],XYt[:,1],s=1,c=c)
    ax.set_aspect('equal')
    hw = (xylimt[0,0]-xylimt[1,0])/100
    hl = hw*2
    
    XYPick = XYt[Npicks,:]
    
    for XYPt in XYPick:
        XPt = XYPt[0]
        YPt = XYPt[1]
        ax.scatter(XPt,YPt,edgecolor='g',facecolor='r',s=50)
    
    ax.plot(XYPick[:,0],XYPick[:,1],c='r',linewidth=2,linestyle='--')
    
    TMt = [None,transMs[hh]]
    at = [0.8,0.5]
    axs_t = [ax,ax_twin]
    for hhh in range(len(axs_t)):
        if hhh==1:
            ls = '--'
        else:
            ls = '-'
        XYaxist = draw_axis(axs_t[hhh],xylimt,TMt[hhh],at[hhh],c=colors_axis[hh],ls=ls)
        if flag_mark[hh] and hhh==1:
            ptx = XYaxist[1,:]
            ptx_1 = ptx*1.5
            ptx_11 = ptx_1.copy()
            ptx_11[1] -=0.3
            ptx_2 = np.matmul(ptx_11,get_rotM(-rotNote_rad/oA))*1.0
            hw = np.linalg.norm(ptx_1)*0.05
            xy_rn = np.zeros([rotNote_div,2])
            cts = 0
            for tht in np.linspace(-rotNote_rad,rotNote_rad,rotNote_div):
                rotMt_tht = get_rotM(tht)
                xy_rn[cts,:] = np.matmul(ptx_1,rotMt_tht)
                cts += 1
            axs_t[hhh].plot(xy_rn[:,0],xy_rn[:,1],c=noteColor)
            axs_t[hhh].arrow(xy_rn[1,0],xy_rn[1,1],
                 xy_rn[0,0]-xy_rn[1,0],xy_rn[0,1]-xy_rn[1,1],
                 color=noteColor,head_width=hw,head_length=hw*2)
            axs_t[hhh].text(ptx_2[0],ptx_2[1],'旋转',color=noteColor,
                 rotation=-np.arctan(ptx[0]/ptx[1])/np.pi*180,
                 fontname='Simhei')
            
            
            pty = XYaxist[-1,:]
            pty_1 = pty*1.1
            pty_11 = pty_1.copy()
            pty_11[0] += 0.4
            pty_2 = np.matmul(pty_11,get_rotM(-rotNote_rad/oA))*1.05
            xy_sn = np.zeros([2,2])
            cts = 0
            for tht in [-rotNote_rad,rotNote_rad]:
                rotMt_tht = get_rotM(tht)
                xy_sn[cts,:] = np.matmul(pty_1,rotMt_tht)
                cts += 1
            axs_t[hhh].plot(xy_sn[:,0],xy_sn[:,1],c=noteColor)
            for tt in [xy_sn,xy_sn[-1::-1,:]]:
                axs_t[hhh].arrow(tt[1,0],tt[1,1],tt[0,0]-tt[1,0],tt[0,1]-tt[1,1],
                     color=noteColor,head_width=hw,head_length=hw*2)
            axs_t[hhh].text(pty_2[0],pty_2[1],'拉伸',color=noteColor,
                 rotation=-np.arctan(pty[0]/pty[1])/np.pi*180,
                 rotation_mode='anchor',verticalalignment='center',
                 fontname='Simhei')
 

    count += 1

for ax_temp in [ax_0,ax_1,ax_mid]:
    for sp in ['left','right','bottom','top']:
        ax_temp.spines[sp].set_visible(False)
        ax_temp.set_xticks([])
        ax_temp.set_xticklabels([])
        ax_temp.set_yticks([])
        ax_temp.set_yticklabels([])



for axt in axs:
    xlimt = np.array(axt.get_xlim())
    ylimt = np.array(axt.get_ylim())
    xlt = np.max(np.abs(np.array(xlimt)))*1.15
    ylt = np.max(np.abs(np.array(ylimt)))*1.15
    axt.set_xlim([-xlt,xlt])
    axt.set_ylim([-ylt,ylt])

st = 100
xst = np.array([-st,0,0,st,0,0,-st,-st])
yst = np.array([st/2,st/2,st,0,-st,-st/2,-st/2,st/2])
ax_mid.plot(xst,yst,c=noteColor,lw=2)
ax_mid.set_aspect('equal')
#figt.tight_layout()

mysavefig(figt)
