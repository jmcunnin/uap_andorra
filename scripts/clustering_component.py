import csv, os
from collections import defaultdict
from itertools import combinations
from math import factorial as f
from random import random

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from scipy.optimize import minimize

from get_regional_dists import _get_connectivity, _get_jaccard, _get_nationality, _get_phonecost_matrix
from distance_metrics import get_naive_distance, get_learned_metric, get_input_matrices


def AgglomerativeCluster(positions, scores, max_dist):
    distarray = ssd.squareform(scores)

    Z = linkage(distarray, 'average')
    pos_map = dict()
    for pos, value in enumerate(positions):
        pos_map[pos] = value

    clusts = fcluster(Z, max_dist, criterion='distance')
    clustered = defaultdict(set)
    for pos, clust in enumerate(clusts):
        val = pos_map[pos]
        clustered[clust].add(pos_map[pos])

    return [list(s) for s in clustered.values()]


def score_clustering(clustering, towers, scores):
    clust_subscore = 0.0
    for clust in clustering:
        if len(clust) == 1:
            clust_subscore += 100
            continue

        curr_score, curr_ct = 0.0, 0
        for t1, t2 in combinations(clust, 2):
            idx1, idx2 = list(towers).index(t1), list(towers).index(t2)
            curr_score += scores[idx1, idx2]
            curr_ct += 1
        clust_subscore += curr_ct

    size_subscore = sum(len(c) for c in clustering)/len(clustering)
    return clust_subscore/len(clustering) - size_subscore


def compute_naive_clustering(max_dist, dists, pos):
    clustering = AgglomerativeCluster(pos, dists, max_dist)
    score = score_clustering(clustering, pos, dists)
    # score += sum(1 for cl in clustering if len(cl) == 1)
    # score += sum(len(cl) for cl in clustering)/len(clustering)
    return score


def compute_learned_clustering(in_arr, pos, conn_mat, ph_mat, jacc_mat, nat_mat):
    max_dist, conn_p, nat_p, jacc_p, ph_p = in_arr
    learned_mat = (1/(conn_p + nat_p+jacc_p+ph_p))*(conn_p*conn_mat + nat_p*nat_mat+jacc_p*jacc_mat+ph_p*ph_mat)
    clustering = AgglomerativeCluster(pos, learned_mat, max_dist)
    score = score_clustering(clustering, pos, learned_mat)
    return score


def do_naive_clustering():
    dists, pos = get_naive_distance()
    clusterings = dict()
    for i in range(2, 31):
        dist_mat = dists[i]
        res = minimize(compute_naive_clustering, .3, method='nelder-mead', args=(dist_mat, pos))
        cl = AgglomerativeCluster(pos, dist_mat, res.x[0])
        clusterings[i] = cl
        print("Completed day: ", i)
    print("Completed Clustering")
    for key, value in clusterings.items():
        ofile = './clusterings/naive/' + str(key) + '.csv'
        writer = csv.writer(open(ofile, 'w'), delimiter=',')
        writer.writerows(value)

def do_learned_clustering():
    pos, conn_mats, ph_mats, jacc_mats, nat_mats = get_input_matrices()
    clusterings = dict()
    for key in range(2, 31):
        c_mat, p_mat, j_mat, n_mat = conn_mats[key], ph_mats[key], jacc_mats[key], nat_mats[key]
        res = minimize(compute_learned_clustering, [.5, .5, .5, .5, .5], method='nelder-mead',
                       args=(pos, c_mat, p_mat, j_mat, n_mat))
        max_dist, conn_p, nat_p, jacc_p, ph_p = res.x
        learned_mat = (1 / (conn_p + nat_p + jacc_p + ph_p)) * (conn_p * c_mat + nat_p * n_mat + jacc_p * j_mat+ ph_p * p_mat)
        clusterings[key] = AgglomerativeCluster(pos, learned_mat, max_dist)
        print("Completed clustering for day: ", key)
        print(res.x[0])

    for key, value in clusterings.items():
        ofile = './clusterings/learned/' + str(key) + '.csv'
        writer = csv.writer(open(ofile, 'w'), delimiter=',')
        writer.writerows(value)


def do_metric_clustering():
    pos, conn_mats, ph_mats, jacc_mats, nat_mats = get_input_matrices()
    clusterings = dict()

    for i in range(2, 31):
        dist_mat = jacc_mats[i] ## Change to change indiviudal cluster
        res = minimize(compute_naive_clustering, .85, method='nelder-mead', args=(dist_mat, pos))
        cl = AgglomerativeCluster(pos, dist_mat, res.x[0])
        clusterings[i] = cl
        print("Completed day: ", i)


    for key, value in clusterings.items():
        ofile = './clusterings/individual/jaccard/' + str(key) + '.csv' # Change location per metric
        writer = csv.writer(open(ofile, 'w'), delimiter=',')
        writer.writerows(value)

# Conn: .7, phone: .005, nat: .005, jac: .85

def read_clustering(ifile):
    olst = list()
    reader = csv.reader(open(ifile), delimiter=',')
    for row in reader:
        olst.append(row)
    return olst

def compute_consensus_mat(c, j, n, p, tows):
    mat = np.zeros((len(tows), len(tows)))
    for clustering in [c,j,n,p]:
        for cl in clustering:
            for c1, c2 in combinations(cl, 2):
                p1, p2 = tows.index(c1), tows.index(c2)
                mat[p1, p2] += 1
                mat[p2, p1] += 1
                assert np.count_nonzero(mat) > 0
    mat += 1
    mat = 1/mat
    np.fill_diagonal(mat, 0)
    return mat




def get_clusterings():
    conn_dir = './clusterings/individual/connectivity/'
    jacc_dir = './clusterings/individual/jaccard/'
    nat_dir = './clusterings/individual/nationality/'
    phone_dir = './clusterings/individual/phone_cost/'

    day_mats = dict()
    all_towers = None
    for ifile in os.listdir(conn_dir):
        day = int(ifile.replace('.csv', ''))

        conn_clusts = [[int(i) for i in l] for l in read_clustering(conn_dir + ifile)]
        jacc_clusts = [[int(i) for i in l] for l in read_clustering(jacc_dir + ifile)]
        nat_clusts = [[int(i) for i in l] for l in read_clustering(nat_dir + ifile)]
        phone_clusts = [[int(i) for i in l] for l in read_clustering(phone_dir + ifile)]
        if not all_towers:
            all_towers = sorted([t for l in conn_clusts for t in l])
        day_mats[day] = compute_consensus_mat(conn_clusts, jacc_clusts, nat_clusts, phone_clusts, all_towers)

    return all_towers, day_mats

def do_consensus_clustering():
    pos, dists = get_clusterings()
    clusterings = dict()
    for i in range(2, 31):
        dist_mat = dists[i]
        res = minimize(compute_naive_clustering, .5, method='nelder-mead', args=(dist_mat, pos))
        cl = AgglomerativeCluster(pos, dist_mat, res.x[0])
        clusterings[i] = cl
        print("Completed day: ", i)
    print("Completed Clustering")
    for key, value in clusterings.items():
        ofile = './clusterings/consensus/' + str(key) + '.csv'
        writer = csv.writer(open(ofile, 'w'), delimiter=',')
        writer.writerows(value)

if __name__ == "__main__":
    do_naive_clustering()
    # do_learned_clustering()
    # do_metric_clustering()
    # do_consensus_clustering()
