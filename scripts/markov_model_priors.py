import numpy as np
import csv
from itertools import product
from collections import defaultdict
from operator import itemgetter


def time_difference(t1, t2):
    t1_conv = 24*60*t1[0] + 60*t1[1] + t1[2] + (1/60) * t1[3]
    t2_conv = 24*60*t2[0] + 60*t2[1] + t2[2] + (1/60) * t2[3]
    return abs(t1_conv - t2_conv)

def filter_paths(input_path, time_limit=1):
    path = [input_path[0][0]]
    last_loc, last_time = input_path[0][0], input_path[0][1]
    for p in input_path[1:]:
        curr_loc, curr_time = p[0], p[1]
        if curr_loc != last_loc:
            path.append(curr_loc)
        elif time_difference(last_time, curr_time) > time_limit:
            path.append(curr_loc)
        last_loc, last_time = curr_loc, curr_time

    return path

def get_paths_byday():
    preader = csv.reader(open("./processed_data/user_paths.csv"), delimiter=',')
    paths = defaultdict(list)

    for row in preader:
        path = [stop.split(';') for stop in row[1:]]
        path = sorted([(int(s[0]), [int(s1) for s1 in s[1:]]) for s in path], key=itemgetter(1))

        first_day = int(path[0][1][0])

        path = filter_paths(path)
        paths[first_day].append(path)

    return paths


def normalize_prob_row(row):
    rcopy = np.copy(row)
    if np.count_nonzero(rcopy) > 0:
        rcopy = rcopy/np.linalg.norm(rcopy)
    return rcopy


def normalize_mat_rows(mat):
    omat_aslst = list()
    for row in mat:
        rcopy = np.copy(row)
        if np.count_nonzero(rcopy) > 0:
            rcopy = rcopy/np.linalg.norm(rcopy)
        omat_aslst.append(rcopy)
    return np.array(omat_aslst)


def generate_transitions(path, wlength):
    if wlength == 1:
        for i in range(len(path) - 1):
            ipath = [path[i]]
            res = path[i+1]
            yield tuple(ipath), res
    elif len(path) > wlength:
        for i in range(len(path) - wlength-1):
            ipath = path[i:i+wlength]
            res = path[i + wlength + 1]
            yield tuple(ipath), res


def generate_transition_matrix(paths, towers, order):
    tpos = dict()
    for i, tower in enumerate(towers):
        tpos[tower] = i
    tower_tuples = [combo for combo in product(towers, repeat=order)]
    ttuple_pos = dict()
    for i, tuple in enumerate(tower_tuples):
        ttuple_pos[tuple] = i

    day_mats = dict()
    for i in range(2, 31):
        day_mats[i] = np.zeros((len(tower_tuples), len(towers)))

    for day, day_paths in paths.items():
        for p in day_paths:
            for ipath, result in generate_transitions(p, order):
                row, column = ttuple_pos.get(ipath, None), tpos.get(result, None)
                if row is not None and column is not None:
                    day_mats[day][row,column] += 1

    omats = dict()
    curr_omat = day_mats[2]
    omats[2] = normalize_mat_rows(curr_omat)

    for i in range(3, 31):
        curr_omat += day_mats[i]
        omats[i] = normalize_mat_rows(curr_omat)

    return omats, tower_tuples


def generate_probability_matrices(paths, towers):
    tower_pos = dict()
    for i, t in enumerate(towers):
        tower_pos[t] = i

    pos_mat = np.zeros((29, len(towers)))
    for day, dpaths in paths.items():
        cday = day-2
        for dpath in dpaths:
            for tower  in dpath:
                tpos = tower_pos.get(tower, None)
                if tpos is not None:
                    pos_mat[cday, tpos] += 1

    olst = list()
    curr_sum = pos_mat[0]
    olst.append(normalize_prob_row(curr_sum))

    for row in pos_mat[1:]:
        curr_sum += row
        olst.append(normalize_prob_row(curr_sum))

    return olst


def generate_priors(order=1):
    treader = csv.reader(open("./processed_data/char_towers_only.csv"), delimiter=',')
    towers = sorted([row for row in treader][0])
    towers = list(map(int, towers))

    paths = get_paths_byday()
    mats, tower_tuples = generate_transition_matrix(paths, towers, order=order)
    mat = generate_probability_matrices(paths, towers)
    return paths, mat, mats, towers, tower_tuples
