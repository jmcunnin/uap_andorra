import numpy as np
from itertools import combinations

from get_regional_dists import _get_connectivity, _get_jaccard, _get_nationality, _get_phonecost_matrix


def _averageForSymmetry(arr):
    for i,j in combinations(range(0, arr.shape[0]), 2):
        v1, v2 = arr[i,j], arr[j,i]
        avg = float(v1+v2)/2.0
        arr[i,j] = avg
        arr[j,i] = avg
    return arr


def get_learned_metric(c_param, p_param, j_param, n_param):
    distances = dict()

    tower_header, c_mats = _get_connectivity()
    _, p_mats = _get_phonecost_matrix()
    _, j_mats = _get_jaccard()
    nationalities, n_mats = _get_nationality(tower_header)

    for key, value in c_mats.items():
        distances[key] = (1/(c_param+ p_param+j_param+n_param))*(c_param * value + p_param*p_mats[key] + j_param * j_mats[key] + n_param * n_mats[key])

    return distances, tower_header


def get_naive_distance():
    distances = dict()

    tower_header, c_mats = _get_connectivity()
    _, p_mats = _get_phonecost_matrix()
    _, j_mats = _get_jaccard()
    nationalities, n_mats = _get_nationality(tower_header)

    for key, value in c_mats.items():
        distances[key] = .25*(_averageForSymmetry(value) + _averageForSymmetry(p_mats[key]) + _averageForSymmetry(j_mats[key]) + _averageForSymmetry(n_mats[key]))

    return distances, tower_header

def avg_all_forsym(mats):
    odct = dict()
    for key, value in mats.items():
        odct[key] = _averageForSymmetry(value)
    return odct

def get_input_matrices():
    distances = dict()

    tower_header, c_mats = _get_connectivity()
    _, p_mats = _get_phonecost_matrix()
    _, j_mats = _get_jaccard()
    nationalities, n_mats = _get_nationality(tower_header)



    return tower_header, avg_all_forsym(c_mats), avg_all_forsym(p_mats), avg_all_forsym(j_mats), avg_all_forsym(n_mats)

