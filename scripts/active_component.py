import csv
from collections import defaultdict
from operator import itemgetter
from ActiveLearner import ActiveLearner
from math import sqrt
import multiprocessing as mp
import time
from functools import partial

def uniqify_path(path):
    olst, seen = list(), set()
    for stop in path:
        if not stop[0] in seen:
            seen.add(stop[0])
            olst.append(stop[0])
    return olst


def import_paths():
    pfile = './processed_data/user_paths.csv'
    reader = csv.reader(open(pfile), delimiter=',')
    dct = defaultdict(list)

    for row in reader:
        user, paths = row[0], row[1:]
        paths = [p.split(';') for p in paths]
        paths = [(int(p[0]), [int(p1) for p1 in p[1:]]) for p in paths]
        dct[user].extend(paths)

    odct = dict()
    for user, paths in dct.items():
        odct[user] = sorted(paths, key=itemgetter(1))

    return odct

def import_user_nats():
    nfile = './processed_data/user_nat_ph.csv'
    reader = csv.reader(open(nfile), delimiter=',')
    nat_dct = dict()

    for row in reader:
        user, nat = row[0], int(row[1])
        nat_dct[user] = nat

    return nat_dct

def mean_and_dev(lst):
    mean = sum(lst)/len(lst)
    std = sqrt(sum((x-mean)**2 for x in lst)/len(lst))
    return mean, std


def parallel_worker(day, pn_tuples, active_learner):
    print("Begun work on day: ", day, " at: ", time.time())
    lst = [ x for x in pn_tuples if len(x[0]) > 5]
    precision_dct = defaultdict(list)
    for elt in lst:
        path, natl = elt
        for i in range(1, 6):
            seen, rest = path[:i], set(path[i:])
            olst = active_learner.get_knn(natl, seen, 20)
            opred = set(olst).difference(set(seen))
            precision_dct[i-1].append(len(rest.intersection(opred))/len(rest))
    return day, [(mean_and_dev(precision_dct[i-1])) for i in range(1,6)]


def do_active_learning():
    paths, nats = import_paths(), import_user_nats()

    print(len(paths), len(nats))

    users_byday = defaultdict(set)
    for user, path in paths.items():
        day = path[0][1][0]
        users_byday[day].add(user)
    print("Preprocess complete")

    print(len(users_byday))

    al = ActiveLearner()

    for i in range(1, 21):
        users = users_byday.get(i, set())
        for user in users:
            path, nat = uniqify_path(paths[user]), nats.get(user, None)
            al.add_path(nat, path)

    tuples_byday = defaultdict(list)
    for i in range(21, 31):
        users = users_byday.get(i, set())
        for user in users:
            path, nat = uniqify_path(paths[user]), nats.get(user, None)
            tuples_byday[i].append((path, nat))

    """
    for i in range(21, 31):
        tups = tuples_byday.get(i, list())
        print(parallel_worker(tups, al))
    """
    ptl_worker = partial(parallel_worker, active_learner=al)
    pool = mp.Pool(processes=3)
    results = [pool.apply_async(ptl_worker, args=(i, tuples_byday.get(i, list()))) for i in range(21, 31)]
    output_intermediate = [r.get() for r in results]
    output = sorted(output_intermediate, key=itemgetter(0))

    ofile = './path_pred_results.csv'
    writer = csv.writer(open(ofile, 'w'), delimiter=',')
    for row in output:
        writer.writerow(row)


if __name__ == "__main__":
    do_active_learning()