from markov_model_priors import *
from functools import partial
from math import sqrt


def mn_std(lst):
    mn = sum(lst)/len(lst)
    std = sqrt((1/len(lst)) * sum((i-mn)**2 for i in lst))
    return mn, std


"""
def compute_path_prob(path, position_mat, tpos_mat):
    prod = 1
    for p in path:
        pos = tpos_mat.get(p, None)
        if pos is not None and position_mat[pos] > 0:
            prod *= position_mat[pos]
    return prod
"""


def compute_transition_precision(day_range, order, tower_tuples, trans_mat, user_paths, pos_mat, towers):
    tpos_dct = dict()
    for i, tow in enumerate(towers):
        tpos_dct[tow] = i

    day_results = list()
    for day in day_range:
        u_paths = user_paths.get(day-1, None)
        u_pos = pos_mat[day-3]
        u_transmat = trans_mat.get(day-1, None)

        ttupos_dct = dict()
        for i, ttups in enumerate(tower_tuples):
            ttupos_dct[ttups] = i

        out = list()
        for p in u_paths:
            for ipath, dest in generate_transitions(p, order):
                ttup_pos = ttupos_dct.get(ipath, None)
                if ttup_pos is not None:
                    trans = list(u_transmat[ttup_pos]) # u_transmat[ttup_pos]
                    tpos = tpos_dct.get(dest, None)
                    if tpos is not None:
                        idx = trans.index(max(trans))
                        if idx == tpos:
                            out.append(1)
                        else:
                            out.append(0)
                        #out.append(trans[tpos])
        m, std = mn_std(out)
        day_results.append((day, m, std))
    print("Completed computation for order: ", order)
    return day_results




def run_markov_model(max_order=5, all_orders=True):
    if not all_orders or max_order == 1:
         user_paths, pos_matrix, transition_matrices, towers, tower_tuples = generate_priors(max_order)
         res = compute_transition_precision(range(20, 31), max_order, tower_tuples, transition_matrices, user_paths, pos_matrix, towers)
         print(res)
    else:
        user_paths, pos_matrix, transition_matrices, towers, tower_tuples = generate_priors(1)

        part_compute = partial(compute_transition_precision, user_paths=user_paths, pos_mat=pos_matrix, towers=towers)

        ordered_trans_mats = [(1, transition_matrices, tower_tuples)]
        for order in range(2, max_order + 1):
            omats, ttups = generate_transition_matrix(user_paths, towers, order=order)
            ordered_trans_mats.append((order, omats, ttups))
            print("Computed priors for ", order)

        results = [part_compute(range(20, 31), o, tt, trans) for o, trans, tt in ordered_trans_mats]
        writer = csv.writer(open("./processed_data/mmperf_ord5_r20_30.csv", 'w'), delimiter=',')
        writer.writerows(results)



if __name__ == "__main__":
    run_markov_model(max_order=3, all_orders=True)