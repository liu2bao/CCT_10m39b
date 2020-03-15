import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

KEY_COOR = 'coor'
KEY_TYPE = 'type'
KEY_CONN = 'conn'
KEY_NO = 'no'
KEY_DIRE = 'direction'
KEY_COLOR = 'color'
DEFAULT_SCALE = 20.0
NDIV_LINE = 30
TYPE_SB_GEN = 1
TYPE_SB_LOAD = -1
TYPE_LINE_AC = 0
TYPE_LINE_TRANS = 1
# DEFAULT_COLORS_G, DEFAULT_COLORS_L, DEFAULT_COLORS_A, DEFAULT_COLORS_T, DEFAULT_COLORS_F = 'k', 'k', 'k', 'k', 'k'
DEFAULT_COLORS_G, DEFAULT_COLORS_L, DEFAULT_COLORS_A, DEFAULT_COLORS_T, DEFAULT_COLORS_F = 'r', 'g', 'k', 'b', 'r'
DEFAULT_COLORS_LINE = {TYPE_LINE_AC:DEFAULT_COLORS_A,TYPE_LINE_TRANS:DEFAULT_COLORS_T}
DEFAULT_COLORS_SB = {TYPE_SB_GEN:DEFAULT_COLORS_G,TYPE_SB_LOAD:DEFAULT_COLORS_L}

xsin_ori = np.linspace(0,2*np.pi,100)
xsin = xsin_ori/np.pi - 1
ysin = np.sin(xsin_ori)
rtc = np.exp(2*np.pi/3*1j)

kthunder = 0.5
xthunder = np.array([-0.25,0.25,0.25,0.75])
ythunder = np.array([xthunder[0]*kthunder,xthunder[1]*kthunder,
                     xthunder[0]*kthunder,xthunder[1]*kthunder])
xythunder = np.vstack([xthunder,ythunder]).T


# For legends
key_right = 'right'
key_left = 'left'
key_center = 'center'
key_lower = 'lower'
key_upper = 'upper'
key_vertical = 'vertical'
key_horizontal = 'horizontal'
label_bus = 'bus'
label_acline = 'acline'
label_transformer = 'transformer'
label_generator = 'generator'
label_load = 'load'
label_fault = 'fault'
dict_trans_ch = {
    label_bus: '母线',
    label_acline: '交流线',
    label_transformer: '变压器',
    label_generator: '发电机',
    label_load: '负荷',
    label_fault: '故障'
}


# Single Line Diagram Drawer
class SLGdrawer(object):
    def __init__(self, buses, special_buses, lines, ax, short_circuits=None,
                 scale_bus = DEFAULT_SCALE/2):
        self.__buses = buses
        self.__special_buses = special_buses
        self.__lines = lines
        self.__ax = ax
        self.__lw_line = 1
        self.__sbdrawers = {1:self.plot_generator,-1:self.plot_load}
        self.__linedrawers = {0:self.plot_acline,1:self.plot_transformer}
        self.__dict_counts_lines = {}
        self.__bus_directions = {}
        self.spec_color = True
        self.scale_bus = scale_bus
        self.text_bus = False
        self.alpha_margin_line = 1
        self.__init_counts_directions_colors()

    @property
    def lw_line(self):
        return self.__lw_line

    @lw_line.setter
    def lw_line(self, lw):
        self.__lw_line = lw

    def __init_counts_directions_colors(self):
        for line in self.__lines:
            connt = line[KEY_CONN]
            kc = (connt[0],connt[1])
            kc_inv = (connt[1],connt[0])
            kt = self.__dict_counts_lines.keys()
            if kc in kt:
                self.__dict_counts_lines[kc] += 1
            elif kc_inv in kt:
                if kc in kt:
                    self.__dict_counts_lines[kc] += 1
                else:
                    self.__dict_counts_lines[kc] = 1
            else:
                self.__dict_counts_lines[kc] = 1
        for bus_no,bus in self.__buses.items():
            if KEY_DIRE not in bus.keys():
                bus[KEY_DIRE] = np.array([1,0])
            else:
                bus[KEY_DIRE] = bus[KEY_DIRE]/np.linalg.norm(bus[KEY_DIRE])

            if KEY_COLOR not in bus.keys():
                bus[KEY_COLOR] = 'k'
            self.__buses[bus_no] = bus

    def plot_generator(self,coor,direction=(0,-1),scale=DEFAULT_SCALE,lw=1,c=DEFAULT_COLORS_G):
        if not self.spec_color:
            c = DEFAULT_COLORS_G
        r = scale/2
        xs = r*0.8
        ys = r*0.5
        dr = np.array(direction)
        dru = dr/np.linalg.norm(dr)
        coor_end = coor+dru*scale
        coor_cen = coor_end+dru*r
        self.__ax.plot([coor[0], coor_end[0]], [coor[1], coor_end[1]], lw=lw, c=c)
        circle_t = Circle(coor_cen,r,lw=lw,facecolor='none',edgecolor=c)
        self.__ax.add_patch(circle_t)
        self.__ax.plot(xs*xsin+coor_cen[0],ys*ysin+coor_cen[1],lw=lw,c=c)

    def plot_load(self,coor,direction=(0,-1),scale=DEFAULT_SCALE,lw=1,c='g'):
        if not self.spec_color:
            c = 'g'
        dr = np.array(direction)
        dru = dr/np.linalg.norm(dr)
        coor_end = coor+dru*scale
        self.__ax.plot([coor[0],coor_end[0]],[coor[1],coor_end[1]],lw=lw,c=c)
        self.plot_et(coor_end,direction,lw=lw,scale=scale/2,c=c)

    def plot_et(self, coor_s, direction=(0, -1), scale=2., lw=1, c='g'):
        R = scale/np.sqrt(3)
        dr = np.array(direction)
        dru = dr / np.linalg.norm(dr)
        coor_c = coor_s+dru*R/2
        drc_0 = np.complex(dru[0],dru[1])
        drc_1 = drc_0*rtc
        drc_2 = drc_1*rtc
        drcs = [drc_0,drc_1,drc_2,drc_0]
        coors = None
        for drc in drcs:
            dru_t = np.array([np.real(drc),np.imag(drc)])
            coor_t = coor_c + R * dru_t
            if coors is None:
                coors = coor_t
            else:
                coors = np.vstack([coors,coor_t])
        self.__ax.plot(coors[:,0],coors[:,1],lw=lw,c=c)

    def plot_bus(self,coor,scale=DEFAULT_SCALE/2,lw=1,dr=np.array([1,0]),c='k',text=None):
        dru = dr/np.linalg.norm(dr)
        ps = coor-dru*scale/2
        pe = coor+dru*scale/2
        if not self.spec_color:
            c = 'k'
        # self.__ax.scatter(coor[0],coor[1],s=scale/3,linewidths=lw)
        self.__ax.plot([ps[0],pe[0]],[ps[1],pe[1]],lw=lw*3,c=c)
        if isinstance(text,str):
            self.__ax.text(coor[0]+5,coor[1]+5,text,fontdict={'family':'Times New Roman'})

    def plot_acline(self,coor_a,coor_b,lw=1,c='k',ndiv=NDIV_LINE,scale=None,trans=False):
        if not self.spec_color:
            c = 'k'
        if trans:
            lw_line = lw
        else:
            lw_line = lw*self.__lw_line
        if (not isinstance(c,str)) and len(c)==2:
            xs = np.linspace(coor_a[0],coor_b[0],ndiv)
            ys = np.linspace(coor_a[1],coor_b[1],ndiv)
            cs = np.linspace(c[0],c[1],ndiv)
            for i in range(ndiv-1):
                coor_a_t = [xs[i],ys[i]]
                coor_b_t = [xs[i+1],ys[i+1]]
                self.plot_acline(coor_a_t,coor_b_t,lw=lw_line,c=cs[i])
        else:
            self.__ax.plot([coor_a[0],coor_b[0]],[coor_a[1],coor_b[1]],lw=lw_line,c=c)

    def plot_transformer(self,coor_a,coor_b,scale=DEFAULT_SCALE,lw=1,c='b',ndiv=NDIV_LINE,prcspec=0.4):
        if not self.spec_color:
            c = 'b'
        r = scale/3
        offset = r*0.4
        dr = coor_b-coor_a
        dru = dr/np.linalg.norm(dr)
        coor_c = (coor_a+coor_b)/2
        coor_o1 = coor_c-dru*offset
        coor_o2 = coor_c+dru*offset
        line_end_1 = coor_o1-dru*r
        line_end_2 = coor_o2+dru*r
        if (not isinstance(c,str)) and len(c)==2:
            func_getc_temp = lambda x: tuple(c[0]*x+c[1]*(1-x))
            color_circle_1 = func_getc_temp(prcspec)
            color_circle_2 = func_getc_temp(1-prcspec)
            color_line1 = [c[0],color_circle_1]
            color_line2 = [color_circle_2,c[1]]
        else:
            color_circle_1 = color_circle_2 = color_line1 = color_line2 = c
        circle1 = Circle(coor_o1,r,facecolor='none',edgecolor=color_circle_1)
        circle2 = Circle(coor_o2,r,facecolor='none',edgecolor=color_circle_2)
        self.plot_acline(coor_a,line_end_1,lw,c=color_line1,ndiv=int(ndiv/2),trans=True)
        self.plot_acline(line_end_2,coor_b,lw,c=color_line2,ndiv=int(ndiv/2),trans=True)
        self.__ax.add_patch(circle1)
        self.__ax.add_patch(circle2)

    def __plot_short_circuits(self,start,dire,scale=DEFAULT_SCALE,c=DEFAULT_COLORS_F,lw=2):
        if dire[0]==0:
            if dire[1]>0:
                theta = np.pi/2
            elif dire[1]<0:
                theta = -np.pi/2
            else:
                theta = 0
        else:
            theta = np.arctan(dire[1]/dire[0])
        rotMt = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        xythunder_t = np.matmul(xythunder,rotMt)*scale+start
        hl_t = scale/4
        hw_t = hl_t/2
        self.__ax.plot(xythunder_t[:,0],xythunder_t[:,1],c=c,lw=lw)
        self.__ax.arrow(xythunder_t[2,0],xythunder_t[2,1],
                        xythunder_t[3,0]-xythunder_t[2,0],xythunder_t[3,1]-xythunder_t[2,1],
                        length_includes_head=True,head_length=hl_t,head_width=hw_t,
                        edgecolor=c,facecolor=c)

    def plot_short_circuit(self,conn_a,conn_b,pos=0.5,scale=DEFAULT_SCALE,offset=None,c=DEFAULT_COLORS_F,coord=False):
        if offset is None:
            offset = np.array([0,0])
        if coord:
            coor_a, coor_b = conn_a, conn_b
        else:
            coor_a = self.__buses[conn_a][KEY_COOR]
            coor_b = self.__buses[conn_b][KEY_COOR]
        coor_sc = (1-pos)*coor_a+pos*coor_b+offset*scale
        dr_ori = coor_b-coor_a
        dr = dr_ori[-1::-1]
        self.__plot_short_circuits(coor_sc,dr,scale=scale,c=c)

    def plot_legend(self,loc=' '.join([key_right,key_center]),orientation=key_vertical,ch=True,offset=None,scale_amp=1):
        scale = self.scale_bus * scale_amp
        width_le_u = scale * 15
        height_le_u = scale * 5
        scale_shift_sym_x = 0.35
        scale_shift_text_y = 0.1
        len_line = scale*4
        direction_ele = np.array([1, 0])
        len_line_v = direction_ele * len_line
        dlen_bus = np.array([len_line / 2, 0])
        dlen_sc = np.array([len_line / 2, scale/2])
        plot_funcs = {
            label_bus: lambda x: self.plot_bus(x + dlen_bus,scale=scale),
            label_acline: lambda x: self.plot_acline(x,x+len_line_v),
            label_transformer: lambda x: self.plot_transformer(x, x + len_line_v,scale=scale*2),
            label_generator: lambda x: self.plot_generator(x, direction_ele,scale=scale*2),
            label_load: lambda x: self.plot_load(x, direction_ele,scale=scale*2),
            label_fault: lambda x: self.__plot_short_circuits(x + dlen_sc,dire=(0,1),lw=1,scale=scale*2)
        }
        types = list(plot_funcs.keys())
        num_types = len(plot_funcs)
        if orientation==key_horizontal:
            width_le = width_le_u * num_types
            height_le = height_le_u
            direction = [1, 0]
        else:
            width_le = width_le_u
            height_le = height_le_u * num_types
            direction = [0, -1]
        direction = np.array(direction)
        if isinstance(loc, str):
            locsub = loc.split()
            xlim_t = self.__ax.get_xlim()
            ylim_t = self.__ax.get_ylim()
            if locsub[0]==key_left:
                loc_x = xlim_t[0]-width_le/2
            elif locsub[0]==key_center:
                loc_x = np.mean(xlim_t)
            else:
                loc_x = xlim_t[1]+width_le/2
            if locsub[1]==key_upper:
                loc_y = ylim_t[1]+height_le/2
            elif locsub[1]==key_lower:
                loc_y = ylim_t[0]-height_le/2
            else:
                loc_y = np.mean(ylim_t)
            loc_center = [loc_x,loc_y]
        else:
            loc_center = loc
        loc_center = np.array(loc_center)
        if offset is not None:
            loc_center += np.array(offset)
        step_t = np.array([width_le_u, height_le_u])
        self.__ax.add_patch(plt.Rectangle(
            xy=loc_center-np.array([width_le,height_le])/2,
            width=width_le, height=height_le,
            edgecolor='grey', facecolor='none'
        ))
        for i in range(num_types):
            loc_ele_t = loc_center+(i-num_types/2+0.5)*direction*step_t
            type_t = types[i]
            if ch:
                label_t = dict_trans_ch[type_t]
                font_t = 'Simhei'
            else:
                label_t = type_t
                font_t = 'Times New Roman'
            loc_ele_sym = loc_ele_t.copy()
            loc_ele_text = loc_ele_t.copy()
            loc_ele_sym[0] -= width_le_u*scale_shift_sym_x
            loc_ele_text[1] -= height_le_u*scale_shift_text_y
            plot_funcs[type_t](loc_ele_sym)
            self.__ax.text(s=label_t,x=loc_ele_text[0],y=loc_ele_text[1],fontname=font_t)

    def plot_slg(self,scale=DEFAULT_SCALE,lw=1):
        dict_count_t = {}
        kt = self.__dict_counts_lines.keys()
        at = self.alpha_margin_line
        for line in self.__lines:
            typet = line[KEY_TYPE]
            connt = line[KEY_CONN]
            kc = (connt[0],connt[1])
            kc_inv = (connt[1],connt[0])
            ktt = dict_count_t.keys()
            if kc in kt:
                count_all_t, conn_a, conn_b = self.__dict_counts_lines[kc], kc[0], kc[1]
            elif kc_inv in kt:
                count_all_t, conn_a, conn_b = self.__dict_counts_lines[kc], kc[1], kc[0]
            else:
                count_all_t, conn_a, conn_b = 1, kc[0], kc[1]
            if kc in ktt:
                count_t = dict_count_t[kc]
            elif kc_inv in ktt:
                count_t = dict_count_t[kc_inv]
            else:
                count_t = 1
                dict_count_t[kc] = 1
            if KEY_COLOR in line.keys():
                color_t = line[KEY_COLOR]
            else:
                color_t = DEFAULT_COLORS_LINE[typet]
            coor_a_ori = self.__buses[conn_a][KEY_COOR]
            coor_b_ori = self.__buses[conn_b][KEY_COOR]
            dire_a = self.__buses[conn_a][KEY_DIRE]
            dire_b = self.__buses[conn_b][KEY_DIRE]
            f = check_same_side(coor_a_ori,coor_b_ori,dire_a,dire_b)
            if not f:
                dire_b = -dire_b
            coor_a = coor_a_ori+((count_t-1+at)/(count_all_t-1+2*at)-0.5)*self.scale_bus*dire_a
            coor_b = coor_b_ori+((count_t-1+at)/(count_all_t-1+2*at)-0.5)*self.scale_bus*dire_b
            ldt = self.__linedrawers[typet]
            ldt(coor_a,coor_b,lw=lw,c=color_t,scale=scale)
            if kc in dict_count_t.keys():
                dict_count_t[kc] += 1
            else:
                dict_count_t[kc] = 1

        for sb in self.__special_buses:
            no_t = sb[KEY_NO]
            type_t = sb[KEY_TYPE]
            direction_t = sb[KEY_DIRE]
            sbd = self.__sbdrawers[type_t]
            coor_t = self.__buses[no_t][KEY_COOR]
            if KEY_COLOR in sb.keys():
                ct = sb[KEY_COLOR]
            else:
                ct = DEFAULT_COLORS_SB[type_t]
            sbd(coor_t,direction_t,scale=scale,lw=lw,c=ct)

        for no,bus_t in self.__buses.items():
            if self.text_bus:
                txtt = str(no)
            else:
                txtt = None
            self.plot_bus(bus_t[KEY_COOR],dr=bus_t[KEY_DIRE],lw=lw,c=bus_t[KEY_COLOR],text=txtt)

        self.__ax.set_aspect('equal')
        self.__ax.spines['left'].set_visible(False)
        self.__ax.spines['right'].set_visible(False)
        self.__ax.spines['top'].set_visible(False)
        self.__ax.spines['bottom'].set_visible(False)
        self.__ax.set_xticks([])
        self.__ax.set_yticks([])


def check_same_side(coor_a,coor_b,dire_a,dire_b):
    coor_diff = coor_b-coor_a
    if coor_diff[0]==0:
        y_a = dire_a[0]
        y_b = dire_b[0]
    else:
        k = coor_diff[1]/coor_diff[0]
        c = coor_a[1]-k*coor_a[0]
        end_a = coor_a+dire_a
        end_b = coor_b+dire_b
        y_a = k*end_a[0]+c-end_a[1]
        y_b = k*end_b[0]+c-end_b[1]
    f = (y_a*y_b)>0
    return f


def get_coordinates():
    path_t, file_t = os.path.split(os.path.abspath(__file__))
    settings_folder = os.path.join(path_t,'settings')
    file_coor_buses = os.path.join(settings_folder, 'coor_buses.txt')
    file_special_buses = os.path.join(settings_folder, 'special_buses.txt')
    file_lines = os.path.join(settings_folder, 'lines.txt')
    # coor_buses = {}
    # special_buses = []
    # lines = []
    with open(file_coor_buses) as f:
        D = f.readlines()
    buses = {}
    for d in D:
        dd = [int(k) for k in d.strip().split()]
        buses[dd[0]] = {KEY_COOR: np.array(dd[1:3])}
        if len(dd) > 4:
            buses[dd[0]][KEY_DIRE] = np.array(dd[3:5])

    with open(file_special_buses) as f:
        D = f.readlines()
    DD = [[int(k) for k in d.strip().split()] for d in D]
    special_buses = [{KEY_NO: dd[0], KEY_TYPE: dd[1], KEY_DIRE: np.array(dd[2:])} for dd in DD]

    with open(file_lines) as f:
        D = f.readlines()

    DD = [[int(k) for k in d.strip().split()] for d in D]
    lines = [{KEY_TYPE: dd[0], KEY_CONN: np.array(dd[1:])} for dd in DD]
    return buses, special_buses, lines


def plot_10m39b_basic(fig=None,return_drawer=False,scale=DEFAULT_SCALE):
    buses, special_buses, lines = get_coordinates()

    if fig is None:
        fig = plt.figure('10M39B',figsize=(7.2,6.8))
    fig.clf()
    ax = fig.gca()
    ax.set_xlim([80,680])
    ax.set_ylim([20,620])
    Drawer = SLGdrawer(buses, special_buses, lines, ax)
    Drawer.spec_color = False
    Drawer.text_bus = True
    Drawer.plot_slg(scale=scale)
    Drawer.plot_short_circuit(4,14,pos=0.2,scale=40,offset=np.array([0,-0.1]))
    fig.show()
    fig.tight_layout()
    if return_drawer:
        return Drawer
    else:
        return fig


# %%
if __name__=='__main__':
    buses, special_buses, lines = get_coordinates()
    
    '''
    import matplotlib.colors as col
    import matplotlib.cm as cm
    import scipy.io as scio

    my_cmap_red = col.LinearSegmentedColormap.from_list('my_cmap', ['#FFD700', '#FF0000'])
    my_cmap_green = col.LinearSegmentedColormap.from_list('my_cmap', ['#E8FFC4','#467500'])
    my_cmap_blue = col.LinearSegmentedColormap.from_list('my_cmap', ['#9393FF','#000079'])
    # extra arguments are N=256, gamma=1.0
    cm.register_cmap(cmap=my_cmap_red)
    cm.register_cmap(cmap=my_cmap_green)

    path_t = os.path.abspath(__file__)
    for i in range(2):
        path_t, file_t = os.path.split(path_t)
    PATH_OUTPUT = os.path.join(path_t,'RESULTS',r'CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate')

    matnamet = 'sorted_P_lines.mat'
    # matnamet = 'sorted_Q_lines.mat'
    matname_bust = 'sorted_Vreal_buses.mat'
    matname_gent = 'sorted_Pg_generators.mat'
    matname_loadt = 'sorted_Pl_loads.mat'

    mat_m = scio.loadmat(os.path.join('..',PATH_OUTPUT,matnamet))
    mat_bus_m = scio.loadmat(os.path.join('..',PATH_OUTPUT,matname_bust))
    mat_gen_m = scio.loadmat(os.path.join('..',PATH_OUTPUT,matname_gent))
    mat_load_m = scio.loadmat(os.path.join('..',PATH_OUTPUT,matname_loadt))
    #lines_d = {tuple(l[KEY_CONN].tolist()):l for l in lines}
    NO_t = mat_m['No_BB_u_sorted']
    NO_all_lines = np.vstack([NO_t[:, 0:2], NO_t[:, 1::-1]])

    w_u_sorted_lines = mat_m['w_u_sorted']
    # w_u_n = (w_u_sorted-np.min(w_u_sorted))/(max(w_u_sorted)-min(w_u_sorted))
    w_u_l_n = w_u_sorted_lines
    w_u_l_n_e = np.vstack([w_u_l_n, w_u_l_n]).flatten()
    w_u_l_n_e = (w_u_l_n_e-np.min(w_u_l_n_e))/(np.max(w_u_l_n_e)-np.min(w_u_l_n_e))
    w_l_map = {tuple(NO_all_lines[i].tolist()):w_u_l_n_e[i] for i in range(w_u_l_n_e.size)}

    lines_c = []
    for l in lines:
        no_t = tuple(l[KEY_CONN].tolist())
        type_t = l[KEY_TYPE]
        if no_t in w_l_map.keys():
            if type_t==0:
                l[KEY_COLOR] = my_cmap_red(w_l_map[no_t])
            elif type_t==1:
                l[KEY_COLOR] = my_cmap_blue(w_l_map[no_t])
        lines_c.append(l)

    w_u_sorted_buses = mat_bus_m['w_u_sorted']
    w_u_sorted_buses = (w_u_sorted_buses-np.min(w_u_sorted_buses))/(np.max(w_u_sorted_buses)-np.min(w_u_sorted_buses))
    NO_all_buses = mat_bus_m['No_BB_u_sorted']
    w_b_map = {int(NO_all_buses[k,0]):float(w_u_sorted_buses[k,0]) for k in range(len(NO_all_buses))}
    for no_b,b in buses.items():
        if no_b in w_b_map.keys():
            b[KEY_COLOR] = my_cmap_red(w_b_map[no_b])
            buses[no_b] = b

    w_u_sorted_gens = mat_gen_m['w_u_sorted']
    w_u_sorted_gens = (w_u_sorted_gens-np.min(w_u_sorted_gens))/(np.max(w_u_sorted_gens)-np.min(w_u_sorted_gens))
    NO_all_gens = mat_gen_m['No_BB_u_sorted']
    w_gen_map = {int(NO_all_gens[k,0]):float(w_u_sorted_gens[k,0]) for k in range(len(NO_all_gens))}

    w_u_sorted_loads = mat_load_m['w_u_sorted']
    w_u_sorted_loads = (w_u_sorted_loads-np.min(w_u_sorted_loads))/(np.max(w_u_sorted_loads)-np.min(w_u_sorted_loads))
    NO_all_loads = mat_load_m['No_BB_u_sorted']
    w_load_map = {int(NO_all_loads[k,0]):float(w_u_sorted_loads[k,0]) for k in range(len(NO_all_loads))}

    special_buses_c = []
    for sb in special_buses:
        typet = sb[KEY_TYPE]
        no_t = sb[KEY_NO]
        if typet==1 and (no_t in NO_all_gens):
            sb[KEY_COLOR] = my_cmap_red(w_gen_map[no_t])
        elif typet==-1 and (no_t in NO_all_loads):
            sb[KEY_COLOR] = my_cmap_green(w_load_map[no_t])
        special_buses_c.append(sb)

    figt = plt.figure('t')
    figt.clf()
    ax = figt.gca()
    ax.set_xlim([80,680])
    ax.set_ylim([20,620])
    #ax.plot([0,500],[0,500],lw=10)
    Drawer = SLGdrawer(buses, special_buses_c, lines, ax)
    Drawer.spec_color = False
    Drawer.text_bus = True
    Drawer.plot_slg()
    Drawer.plot_short_circuit(4,14,pos=0.2,scale=40,offset=np.array([0,-0.1]))
    figt.show()
    figt.tight_layout()
    '''