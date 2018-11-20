import os, csv
from operator import itemgetter
from collections import defaultdict
from math import sqrt

def ordered_uniqify(lst):
    seen = set()
    olst = list()
    for l in lst:
        if l not in seen:
            seen.add(l)
            olst.append(l)
    return olst


def import_clustering_dir(cdir):
    clusters = dict()
    for ifile in os.listdir(cdir):
        day = int(ifile.replace('.csv', ''))
        reader = csv.reader(open(cdir + ifile), delimiter=',')
        day_clustering = [[int(i) for i in row] for row in reader]
        clusters[day] = day_clustering
    return clusters

def import_userpaths():
    fpath = './processed_data/user_paths.csv'
    paths_byday = defaultdict(list)

    reader = csv.reader(open(fpath), delimiter=',')
    for row in reader:
        user, paths = row[0], row[1:]
        paths = [p.split(';') for p in paths]
        paths = [(int(p[0]), [int(i) for i in p[1:]]) for p in paths]
        paths = sorted(paths, key=itemgetter(1))
        day = paths[0][1][0]
        opath = ordered_uniqify([v[0] for v in paths])
        if len(opath) > 1:
            paths_byday[day].append(opath)
    return paths_byday

def compute_pathprecision(path, clusterings):
    info, rest_path = path[0], set(path[1:])

    cl = None
    for clust in clusterings:
        if info in clust:
            cl = set(clust)
            cl.remove(info)
            break

    if cl is None:
        return 0, 0

    return len(rest_path.intersection(cl))/len(rest_path), 1


def compute_pathprecision_multclusts(path, clust1, clust2):
    info, rest_path = path[0], set(path[1:])

    cl = set()
    for clust in clust1:
        if info in clust:
            cl = cl.union(set(clust))
            break

    for clust in clust2:
        if info in clust:
            cl = cl.union(set(clust))
            break

    cl.remove(info)

    if len(cl) is 0:
        return 0, 0

    return len(rest_path.intersection(cl))/len(rest_path), 1



def combine_metrics(cdir, ldir, paths_byday):
    con_clusts = import_clustering_dir(cdir)
    learn_clusts = import_clustering_dir(ldir)

    precision_lst = list()
    for day, paths in paths_byday.items():
        c_clust, l_clust = con_clusts[day], learn_clusts[day]
        precision, count = 0.0, 0
        for path in paths:
            cprec, cct = compute_pathprecision_multclusts(path, c_clust, l_clust)
            precision += cprec
            count += cct
        precision_lst.append((day, precision/count))
    return precision_lst

def evaluate_clustering(cdir, paths_byday):
    clusterings_dir = import_clustering_dir(cdir)

    precision_lst= list()
    for day, clustering in clusterings_dir.items():
        daypaths = paths_byday[day]
        precision, count = 0.0, 0
        for path in daypaths:
            cprec, cct = compute_pathprecision(path, clustering)
            precision += cprec
            count += cct
        precision_lst.append((day, precision/count))
    return precision_lst

def compute_mean_std(lst):
    mean = sum(v[1] for v in lst)/len(lst)
    std = sqrt(sum((x[1]-mean)**2 for x in lst)/len(lst))
    return mean, std


if __name__ == "__main__":
    by_day = import_userpaths()
    print(sum(sum(len(v) for v in v0) for _, v0 in by_day.items())/sum(len(l) for _,l in by_day.items()))

    consensus_dir = './clusterings/consensus/'
    learned_dir = './clusterings/learned/'
    naive_dir = './clusterings/naive/'

    cluster_dirs = [consensus_dir, learned_dir, naive_dir]
    prec_ofile = './clusterings/precision.csv'
    writer = csv.writer(open(prec_ofile, 'w'), delimiter=',')
    lst = list()
    for idir in cluster_dirs:
        precision = evaluate_clustering(idir, by_day)
        orow = [idir]
        orow.extend(precision)
        writer.writerow(orow)
        mean, std = compute_mean_std(precision)
        lst.append([idir, mean, std])
        print("Completed processing of: ", idir)

    precision = combine_metrics(consensus_dir, learned_dir, by_day)
    mean, std = compute_mean_std(precision)

    orow = ['combined_clusterings']
    orow.extend(precision)
    writer.writerow(orow)
    lst.append(["Combined", mean, std])

    writer.writerow(["Precision_Stats", "Mean", "std"])
    for row in lst:
        writer.writerow(row)


