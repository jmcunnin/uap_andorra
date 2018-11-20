import numpy as np
import csv, os
from itertools import combinations
from markov_model_priors import get_paths_byday


def norm_mat(mat):
    olst = list()
    for row in mat:
        rcopy = np.copy(row)
        if np.count_nonzero(rcopy) > 0:
            rcopy = rcopy/np.linalg.norm(rcopy)
        olst.append(rcopy)
    return np.array(olst)


def import_clusterings(idir):
    clusterings = dict()

    for ifile in os.listdir(idir):
        day = int(ifile.replace('.csv', ''))
        reader = csv.reader(open(idir+ifile), delimiter=',')
        clusterings[day] = [[int(t) for t in c] for c in reader]

    towers = sorted(list(i for cl in clusterings[2] for i in cl))
    return clusterings, towers


def convert_to_similarity(clusterings, towers):
    clust_mats = dict()

    tposdct = dict()
    for i, t in enumerate(towers):
        tposdct[t] = i

    for day, clusts in clusterings.items():
        mat = np.zeros((len(towers), len(towers)))
        for clust in clusts:
            for t1, t2 in combinations(clust, 2):
                p1, p2 = tposdct[t1], tposdct[t2]
                mat[p1,p2] = 1
                mat[p2,p1] = 1
            if len(clust) == 1:
                p1 = tposdct[clust[0]]
                mat[p1, p1] = 1
        clust_mats[day] = mat

    return clust_mats


def aggregate_clusterings(clusts):
    agg_clusts = dict()
    curr_clust = clusts[2]
    agg_clusts[2] = norm_mat(curr_clust)

    for day in range(3, 31):
        curr_clust += clusts[day]
        agg_clusts[day] = norm_mat(curr_clust)

    return agg_clusts


def aggregate_two_clusterings(clusts1, clusts2):
    agg_clusts = dict()
    curr_clust = clusts1[2] + clusts2[2]
    agg_clusts[2] = norm_mat(curr_clust)

    for day in range(3, 31):
        curr_clust += clusts1[day]
        curr_clust += clusts2[day]

        agg_clusts[day] = norm_mat(curr_clust)

    return agg_clusts


def pairwise_iterator(lst):
    if len(lst) > 1:
        for i in range(len(lst)-1):
            yield lst[i], lst[i+1]


def normalize_mats(clusts):
    odct = dict()
    for key, mat in clusts.items():
        odct[key] = norm_mat(mat)
    return odct

def consensus_normalize(clusts1, clusts2):
    odct = dict()
    for key, mat in clusts1.items():
        omat = mat + clusts2[key]
        odct[key] = norm_mat(omat)
    return odct


def compute_precision(agg_clusts, upaths, towers, day_range):
    preds_byday = list()
    tpos_dct = dict()
    for i, t in enumerate(towers):
        tpos_dct[t] = i

    for day in day_range:
        agg_clust = agg_clusts[day-1]
        d_upaths = upaths[day]

        preds = list()
        for path in d_upaths:
            for curr, dest in pairwise_iterator(path):
                pos = tpos_dct[curr]
                row = list(agg_clust[pos])
                idx = row.index(max(row))

                if idx == tpos_dct[dest]:
                    preds.append(1)
                else:
                    preds.append(0)

        preds_byday.append((day, sum(preds)/len(preds)))
    return preds_byday


def get_clustering():
    ndir, ldir, cdir = './clusterings/naive/', './clusterings/learned/', './clusterings/consensus/'

    lclusts, towers = import_clusterings(ldir)
    cclusts, _ = import_clusterings(cdir)

    lsim = aggregate_clusterings(convert_to_similarity(lclusts, towers))
    csim = aggregate_clusterings(convert_to_similarity(cclusts, towers))
    lpc = aggregate_two_clusterings(convert_to_similarity(lclusts, towers), convert_to_similarity(cclusts, towers))

    return towers, lsim, csim, lpc



if __name__ == "__main__":
    ndir, ldir, cdir = './clusterings/naive/', './clusterings/learned/', './clusterings/consensus/'

    nclusts, towers = import_clusterings(ndir)
    lclusts, _ = import_clusterings(ldir)
    cclusts, _ = import_clusterings(cdir)

    nsim = convert_to_similarity(nclusts, towers)
    lsim = convert_to_similarity(lclusts, towers)
    csim = convert_to_similarity(cclusts, towers)

    upaths = get_paths_byday()
    """

    naive = compute_precision(aggregate_clusterings(nsim), upaths, towers, range(20, 31))
    learned = compute_precision(aggregate_clusterings(lsim), upaths, towers, range(20, 31))
    consensus = compute_precision(aggregate_clusterings(csim), upaths, towers, range(20, 31))
    lpc = compute_precision(aggregate_two_clusterings(lsim, csim), upaths, towers, range(20, 31))
    """
    naive = compute_precision(normalize_mats(nsim), upaths, towers, range(20, 31))
    learned = compute_precision(normalize_mats(lsim), upaths, towers, range(20,31))
    consensus = compute_precision(normalize_mats(csim), upaths, towers, range(20,31))
    lpc = compute_precision(consensus_normalize(lsim, csim), upaths, towers, range(20,31))


    writer = csv.writer(open("./processed_data/clust_perf_ind_r20_30.csv", 'w'), delimiter=',')
    writer.writerow(['Day', 'Naive', 'Learned', 'Consensus', 'Combined'])
    for n, l, c, lpc in zip(naive, learned, consensus, lpc):
        row = [n[0], n[1], l[1], c[1], lpc[1]]
        writer.writerow(row)

    print("written to outfile")
