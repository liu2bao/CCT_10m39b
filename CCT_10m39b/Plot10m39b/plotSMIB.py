import matplotlib.pyplot as plt
import numpy as np
from CCT_10m39b.Plot10m39b.plot10m39b import SLGdrawer,KEY_COOR,KEY_TYPE,KEY_CONN,KEY_DIRE,KEY_NO


def plot_SMIB_basic(fig=None, return_drawer=False):
    buses = {1: {KEY_COOR: np.array([0, 0]), KEY_DIRE: np.array([0, 1])},
             2: {KEY_COOR: np.array([50, 0]), KEY_DIRE: np.array([0, 1])},
             3: {KEY_COOR: np.array([150, 0]), KEY_DIRE: np.array([0, 1])},
             4: {KEY_COOR: np.array([200, 0]), KEY_DIRE: np.array([0, 1])}}
    special_buses = [{KEY_NO: 1, KEY_TYPE: 1, KEY_DIRE: np.array([-1, 0])},
                     {KEY_NO: 4, KEY_TYPE: -1, KEY_DIRE: np.array([1, 0])}]
    lines = [{KEY_TYPE: 1, KEY_CONN: np.array([1, 2])},
             {KEY_TYPE: 1, KEY_CONN: np.array([3, 4])},
             {KEY_TYPE: 0, KEY_CONN: np.array([2, 3])},
             {KEY_TYPE: 0, KEY_CONN: np.array([2, 3])}]

    if fig is None:
        fig = plt.figure('SMIB', figsize=(5.4, 2))
    fig.clf()
    ax = fig.gca()
    ax.set_xlim([-50, 250])
    ax.set_ylim([-50, 50])
    # ax.plot([0,500],[0,500],lw=10)
    Drawer = SLGdrawer(buses, special_buses, lines, ax)
    Drawer.spec_color = False
    Drawer.alpha_margin_line = 0.5
    Drawer.plot_slg()
    Drawer.plot_short_circuit(2, 3, 0.2, offset=np.array([0, -0.25]))

    dict_text = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    for no_t, txt_t in dict_text.items():
        ax.text(buses[no_t][KEY_COOR][0] - 4, buses[no_t][KEY_COOR][1] + 10, txt_t,
                fontdict={'family': 'Times New Roman', 'size': 16})

    fig.show()
    fig.tight_layout()

    if return_drawer:
        return Drawer
    else:
        return fig

#%%
if __name__=='__main__':
    plot_SMIB_basic()