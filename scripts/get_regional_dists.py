import numpy as np
import os

from itertools import combinations


def _get_phonecost_matrix():
    fpath = "./processed_data/phone_cost.csv"
    mat = np.loadtxt(open(fpath))
    towers = np.array([int(x) for x in mat[0,]][1:])
    mat = mat[2:, 1:]
    omats, day = dict(), 2
    for curr_day in mat:
        omat = list()
        for o in curr_day:
            ilst = list()
            for i in curr_day:
                ilst.append(abs(o-i))
            if np.count_nonzero(np.array(ilst)) > 0:
                omat.append(np.array(ilst)/np.linalg.norm(np.array(ilst)))
            else:
                omat.append(np.array(ilst))
        omats[day] = np.array(omat)
        day += 1
    del omats[31]
    return towers, omats


def _get_jaccard():
    dirpath = './processed_data/jaccard/'
    mat_dir = dict()
    for ifile in os.listdir(dirpath):
        if ifile == '1.csv' or ifile == '31.csv':
            continue
        day = int(ifile.replace('.csv',''))
        mat = np.loadtxt(open(dirpath + ifile))
        tower_lst = np.array([int(x) for x in mat[0,]])
        mat = mat[1:,]
        mat_dir[day] = mat

    return tower_lst, mat_dir


def _get_connectivity():
    dirpath = './processed_data/connectivity/'
    towers, mat_dir = list(), dict()
    for ifile in os.listdir(dirpath):
        if ifile == '1.csv' or ifile == '31.csv':
            continue
        day = int(ifile.replace('.csv', ''))
        mat = np.loadtxt(open(dirpath + ifile))
        tower_lst = np.array([int(x) for x in mat[0,]][1:])
        mat = mat[1:, 1:]
        towers.append(tower_lst)
        olst = list()
        for lst in mat:
            olst.append(1 - lst/np.linalg.norm(lst))
        mat = np.array(olst)
        np.fill_diagonal(mat, 0)
        mat_dir[day] = mat
    return towers[0], mat_dir


def _get_tow_array(mat, tower, towers):
    if tower in towers:
        idx = towers.index(tower)
        return mat[idx,]
    else:
        return np.array([0]*192)


def _get_nationality(towers):
    dirpath = './processed_data/nationality/'
    mat_dir = dict()
    nats = list()
    nat_towers= list()
    for ifile in os.listdir(dirpath):
        if ifile == '1.csv' or ifile == '31.csv':
            continue
        day = int(ifile.replace('.csv', ''))
        mat = np.loadtxt(open(dirpath + ifile))
        nats = np.array([int(x) for x in mat[0,]][1:])
        nat_towers = [int(x) for x in mat[:,0]][1:]

        mat = mat[1:,1:]
        olst = list()
        for t1 in towers:
            ilst = list()
            t1_arr = _get_tow_array(mat, t1, nat_towers)
            for t2 in towers:
                t2_arr = _get_tow_array(mat, t2, nat_towers)
                ilst.append(np.linalg.norm(t1_arr-t2_arr))
            arr = np.array(ilst)
            if np.count_nonzero(arr)>0:
                olst.append(arr/np.linalg.norm(arr))
            else:
                olst.append(arr)
        mat = np.array(olst)
        mat[mat==0] = 1
        np.fill_diagonal(mat, 0)
        mat_dir[day] = mat

    return nats, mat_dir


if __name__ == "__main__":
    ph_towers, phone_mat = _get_phonecost_matrix()
    j_towers, j_mats = _get_jaccard()
    c_towers, c_mats = _get_connectivity()
    nats, n_mats = _get_nationality(ph_towers)


