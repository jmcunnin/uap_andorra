import csv, os
import numpy as np
from itertools import combinations



def import_clusts(idir):
    clusts = dict()
    for ifile in os.listdir(idir):
        day = int(ifile.replace('.csv', ''))
        reader = csv.reader(open(idir + ifile), delimiter=',')
        clusts[day] = [[int(i) for i in cl] for cl in reader]
    return clusts

def clustering_sim(clust1, clust2):
    numerator = sum(max((len(set(l1).intersection(set(l2))) for l2 in clust2)) for l1 in clust1)
    denom1 = sum(len(cl1) for cl1 in clust1)
    denom2 = sum(len(cl2) for cl2 in clust2)
    return numerator/denom1, numerator/denom2


def gen_simmat(clusts1, clusts2):
    sim_lst1 = list()
    sim_lst2 = list()
    for i in range(2, 31):
        cl1, cl2 = clusts1[i], clusts2[i]
        l, r = clustering_sim(cl1, cl2)
        sim_lst1.append(l)
        sim_lst2.append(r)
    print(sum(sim_lst1)/len(sim_lst1))
    print(sum(sim_lst2)/len(sim_lst2))




if __name__ == "__main__":
    con_dir = './clusterings/consensus/'
    lea_dir = './clusterings/learned/'
    nai_dir = './clusterings/naive/'

    jacc_dir = './clusterings/individual/jaccard/'
    conn_dir = './clusterings/individual/connectivity/'
    nat_dir = './clusterings/individual/nationality/'
    pho_dir = './clusterings/individual/phone_cost/'

    con_clusts = import_clusts(con_dir)
    lea_clusts = import_clusts(lea_dir)

    gen_simmat(lea_clusts, con_clusts)

