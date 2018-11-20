import numpy as np
import time
from functools import partial
import heapq
from collections import Counter
import multiprocessing as mp
from operator import itemgetter
from math import ceil
from random import shuffle

from markov_model_priors import generate_priors, generate_transition_matrix
from cspa_clustering import get_clustering


def path_generator(day_paths):
    for path in day_paths:
        if len(path) >= 3:
            for i in range(len(path) - 2):
                two_prior, prior, nxt = path[i:i+3]
                o1, o2 = tuple([prior]), tuple([two_prior, prior])
                yield o1, o2, prior, nxt


def get_xy(paths, tower_pos, o1_tt_pos, o2_tt_pos, o1_curr_mat, o2_curr_mat, le_curr_mat, co_curr_mat):
    for o1, o2, curr, nxt in path_generator(paths):
        o1_prior = o1_curr_mat[o1_tt_pos[o1]]
        o2_prior = o2_curr_mat[o2_tt_pos[o2]]

        le_prior = le_curr_mat[tower_pos[curr]]
        co_prior = co_curr_mat[tower_pos[curr]]

        info_row = np.concatenate((o1_prior, o2_prior, le_prior, co_prior))
        yield info_row, nxt


def splitter_generator(test_paths, split_size):
    curr_pos, total = 0, len(test_paths)
    while curr_pos + split_size < total:
        sidx = curr_pos
        curr_pos += split_size
        yield test_paths[sidx:curr_pos]
    else:
        yield test_paths[curr_pos:]



def knn_worker(test_paths, ref_paths, generator_partial, k=30):
    test_generator = generator_partial(paths=test_paths)
    matches, total = 0, 0
    for curr_vec, nxt_stop in test_generator:
        ref_generator = generator_partial(ref_paths)
        ct, h = 0, list()
        for ref_row, ref_next in ref_generator:
            if ct > k:
                heapq.heappushpop(h, (-np.linalg.norm(ref_row - curr_vec), ref_next))
            else:
                heapq.heappush(h, (-np.linalg.norm(ref_row - curr_vec), ref_next))
            ct += 1
        if Counter((i[1] for i in h)).most_common(1)[0][0] == nxt_stop:
            matches += 1
        total += 1
    return matches, total


def do_logregress():
    cspa_towers, learned_clusts, consensus_clusts, combined_clusts = get_clustering()
    upaths, _, o1_priors_mats, markov_towers, o1_tower_tuples = generate_priors()
    o2_priors_mats, o2_tower_tuples = generate_transition_matrix(upaths, markov_towers, order=2)

    tower_pos = dict()
    for i, tow in enumerate(cspa_towers):
        tower_pos[tow] = i

    o1_tt_pos = dict()
    for i, ttup in enumerate(o1_tower_tuples):
        o1_tt_pos[ttup] = i

    o2_tt_pos = dict()
    for i, ttup in enumerate(o2_tower_tuples):
        o2_tt_pos[ttup] = i

    dlength = len(upaths[22])
    print("Need to test paths: ", len(upaths[22]))

    o1_curr_mat = o1_priors_mats[21]
    o2_curr_mat = o2_priors_mats[21]
    le_curr_mat = learned_clusts[21]
    co_curr_mat = consensus_clusts[21]

    paths = [path for day in range(2, 22) for path in upaths[day]]

    info_gen_part = partial(get_xy, tower_pos=tower_pos, o1_tt_pos=o1_tt_pos, o2_tt_pos=o2_tt_pos,
                            o1_curr_mat=o1_curr_mat, o2_curr_mat=o2_curr_mat, le_curr_mat=le_curr_mat,
                            co_curr_mat =co_curr_mat)

    p = mp.Pool(processes=6)
    chunks = ceil(dlength/6)*6
    knn_partial = partial(knn_worker, ref_paths=paths, generator_partial=info_gen_part, k=30)
    print("Starting call to knn worker")
    istart = time.time()
    #knn_partial(upaths[22])
    results = [p.apply_async(knn_partial, args=(ipath,)) for ipath in splitter_generator(upaths[22][:140], 25)]
    matched = [r.get() for r in results]
    mtch, total = sum(i[0] for i in matched), sum(i[1] for i in matched)
    print("Accuracy was: ", mtch/total, total)
    print("Completed working for one, time: ", time.time()-istart)


if __name__ == "__main__":
    do_logregress()


