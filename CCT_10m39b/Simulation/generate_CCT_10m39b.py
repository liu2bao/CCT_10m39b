import sys

sys.path.append(r'F:\Programs\PythonWorks\PowerSystem\PyPSASP')

from PyPSASP.PSASPClasses.PSASP import PSASP, CCT_generator
from PyPSASP.constants import const
import numpy as np
import random
import os
import time

PATH_TEMP = r'E:\01_Research\98_Data\IEEE10m39b\PSASP\Temp_AVR_GOV_PSS'
PATH_TEMP_TOPO = r'E:\01_Research\98_Data\IEEE10m39b\PSASP\Temp_AVR_GOV_PSS_topoChange'
PATH_OUTPUT = os.path.join('RESULTS', 'CCT_RESULTS_fixBV_fixPlrate_fixPlQlrate_topoChange')


def func_change_lfs(P):
    if isinstance(P, PSASP):
        gen_ori = P.parser.parse_single_s_lfs(const.LABEL_GENERATOR)
        load_ori = P.parser.parse_single_s_lfs(const.LABEL_LOAD)
        gen_new = gen_ori.copy()
        load_new = load_ori.copy()
        Psum = 0
        for hh in range(len(gen_new)):
            gen_new[hh][const.GenPgKey] = gen_new[hh][const.PmaxKey] * (random.random() * 0.5 + 0.5)
            gen_new[hh][const.V0Key] = (random.random() * 0.2 + 0.95)
            Psum = Psum + gen_new[hh][const.GenPgKey]
        rands_t = [random.random() for hh in range(len(load_new))]
        Ap = random.random() * 0.3 + 0.8
        Pls_t = [x / sum(rands_t) * Ap * Psum for x in rands_t]
        for hh in range(len(load_new)):
            # load_new[hh][const.LoadQlKey] = load_ori[hh][const.LoadQlKey]/load_ori[hh][const.LoadPlKey]*Pls_t[hh]
            load_new[hh][const.LoadPlKey] = Pls_t[hh]
            # load_new[hh][const.LoadQlKey] = load_ori[hh][const.LoadQlKey] * ((random.random() - 0.5) * 2.4)
        P.writer.write_to_file_s_lfs_autofit(gen_new)
        P.writer.write_to_file_s_lfs_autofit(load_new)


def func_change_lfs_simple_2(P):
    if isinstance(P, PSASP):
        gen_ori = P.parser.parse_single_s_lfs(const.LABEL_GENERATOR)
        load_ori = P.parser.parse_single_s_lfs(const.LABEL_LOAD)
        gen_new = gen_ori.copy()
        load_new = load_ori.copy()

        Pgmax = np.array([g[const.PmaxKey] for g in gen_ori])
        Pg_ori = np.array([g[const.GenPgKey] for g in gen_ori])
        Pl_ori = np.array([l[const.LoadPlKey] for l in load_ori])
        Ql_ori = np.array([l[const.LoadQlKey] for l in load_ori])
        Ng = len(gen_new)
        Nl = len(load_new)
        Pg_new = (np.random.rand(Ng) * 0.5 + 0.5) * Pgmax
        Psum_new = np.sum(Pg_new)
        Psum_ori = np.sum(Pg_ori)
        Pl_new = Pl_ori / Psum_ori * Psum_new
        Ql_new = Ql_ori / Psum_ori * Psum_new
        for hh in range(Ng):
            gen_new[hh][const.GenPgKey] = Pg_new[hh]
        for hh in range(Nl):
            load_new[hh][const.LoadPlKey] = Pl_new[hh]
            load_new[hh][const.LoadQlKey] = Ql_new[hh]
        P.writer.write_to_file_s_lfs_autofit(gen_new)
        P.writer.write_to_file_s_lfs_autofit(load_new)


def func_change_lfs_simple(P, list_idx_G=(1, -3)):
    if isinstance(P, PSASP):
        gen_ori = P.parser.parse_single_s_lfs(const.LABEL_GENERATOR)
        gen_new = gen_ori.copy()
        for idx_G in list_idx_G:
            gen_new[idx_G][const.GenPgKey] = gen_new[idx_G][const.PmaxKey] * (random.random() * 0.1 + 0.85)
        P.writer.write_to_file_s_lfs_autofit(gen_new)


count_topo_all = 0
change_topo = False
Pgl = PSASP(PATH_TEMP_TOPO)
NO_L_ST = 12


def get_sub_topo_num(cta,numa):
    solt = ((2 * numa + 1) - np.sqrt((2 * numa + 1) ** 2 - 8 * cta)) / 2
    ct1 = int(np.ceil(solt))
    cp = int((2 * numa - ct1 + 2) * (ct1 - 1) / 2)
    ct2 = cta - cp + ct1 - 1
    return ct1, ct2


def func_change_lfs_simple_2_topo(P):
    global count_topo_all
    if isinstance(P, PSASP):
        LF2_t = P.parser.parse_single_s_lfs(const.LABEL_ACLINE)
        numa = len(LF2_t)
        for hh in range(len(LF2_t)):
            LF2_t[hh][const.MarkKey] = 1
        if count_topo_all != 0:
            while True:
                count_topo_1, count_topo_2 = get_sub_topo_num(count_topo_all,numa)
                if any([k==NO_L_ST for k in [count_topo_1,count_topo_2]]):
                    count_topo_all += 1
                else:
                    break
            LF2_t[count_topo_1 - 1][const.MarkKey] = 0
            LF2_t[count_topo_2 - 1][const.MarkKey] = 0
        P.writer.write_to_file_s_lfs(const.LABEL_ACLINE, LF2_t)
        if count_topo_all == 0:
            func_change_lfs_simple_2(P)
        count_topo_all += 1
        if count_topo_all > (numa * (numa + 1) / 2):
            count_topo_all = 0


# Pg2 = PSASP(r'E:\01_Research\98_Data\IEEE10m39b\PSASP\t')
# for i in range(100):
#     func_change_lfs_simple_2_topo(Pg2)
#     LF2_temp = Pg2.parser.parse_single_s_lfs(const.LABEL_ACLINE)
#     M = [a[const.MarkKey] for a in LF2_temp]
#     print(str(count_topo_all) + ' : ' + str(M))

if __name__ == '__main__':
    # time.sleep(2000)
    CCT_generator = CCT_generator(path_temp=PATH_TEMP_TOPO,
                                  path_output=PATH_OUTPUT,
                                  func_change_lfs=func_change_lfs_simple_2_topo)
    countT = 0
    while countT < 5000:
        CCT_generator.run_sim_CCT_once()
        countT += 1
        time.sleep(3)
