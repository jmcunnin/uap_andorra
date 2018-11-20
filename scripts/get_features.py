import os, csv
from collections import namedtuple, defaultdict, Counter
from helper_funcs import read_pointers, jaccard_fn, get_csv_files
from itertools import combinations
import numpy as np

"""
The out-directory path to be written, directory written as follows:
        -- users: A directory containing, for each tower, a list of tuples: [user, dd, hh, mm, ss]
        -- nationality: A directory containing the daily pairwise distance matrices
        -- phone_cost.csv: A CSV containing the daily average phone cost of each region
        -- jaccard: A directory containing the daily pairwise jaccard distance matrices
        -- connectivity: A directory containing the daily pairwise connectivity distance matrices

Each of these folders,  contains a set of matrices extracting each feature. For further info, see the individual functions.

"""
ODirPaths = namedtuple('ODirPaths', 'users nats jaccs conns costs')
ReferencePointers = namedtuple('ReferencePointers', 'char_towers char_countries')


def construct_odir(odir):
    """
    Constructs the out-directory as described in parse_data_input
    :param odir: The filepath of the out directory to be written
    :exception ValueError: Raises a value error where odir already exists
    """
    odirs = ODirPaths(users=odir + '/users/',
                      nats=odir + '/nationality/',
                      jaccs=odir + '/jaccard/',
                      conns=odir + '/connectivity/',
                      costs=odir + '/phone_cost.csv')
    for feature_dir in odirs[1:-1]:
        os.makedirs(feature_dir)

    return odirs


def compute_jaccard_feature(idir, odir, days):
    """
    Computes the jaccard feature for each tower and outputs to numpy matrices
    :param idir: the input directory of user files
    :param odir: the output directory
    :param days: number of days in the month
    :outputs: modifies the output directory with daily pairwise distance functions
    """
    users = dict()
    towers, position = dict(), 0
    for ifile in get_csv_files(idir):
        tower = int(ifile.replace('.csv', ''))
        towers[tower] = position # Maps to position in eventual matrix
        position += 1

        tdict = defaultdict(set)

        reader = csv.reader(open(idir + '/' + ifile), delimiter=',')
        for row in reader:
            user, day = row[0], int(row[1])
            tdict[day].add(user)

        users[tower] = tdict

    tower_lst_ps = [(k,v) for k,v in towers.items()]
    tower_lst_ps.sort(key=lambda x:x[1])
    towers_lst = [p[0] for p in tower_lst_ps]

    num_tows = len(towers.keys())
    for day in range(1, days + 1):
        ofile = odir + str(day) + '.csv'
        mat = np.zeros((num_tows, num_tows))
        for t1, t2 in combinations(towers.keys(), 2):
            p1, p2 = towers.get(t1, 0), towers.get(t2, 0)
            s1 = users[t1].get(day, set())
            s2 = users[t2].get(day, set())
            jacc = jaccard_fn(s1, s2)
            mat[p1, p2] = jacc
            mat[p2, p1] = jacc

        omat = np.append([towers_lst], mat, axis=0)
        np.savetxt(ofile, omat)


def compute_phonecosts(idir, ofile, days):
    """
    Compute the per-tower average phone costs
    :param idir: the input directory of user data
    :param ofile: the output file to write to
    :param days: number of days in the month
    :return: writes the average phone cost in each tower per day.
        The top row is the list of towers
    """
    print(ofile)
    users = dict()
    towers, position = dict(), 0
    for ifile in get_csv_files(idir):
        tower = int(ifile.replace('.csv', ''))
        towers[tower] = position # Maps to position in eventual matrix
        position += 1

        tdict = defaultdict(list)

        reader = csv.reader(open(idir + '/' + ifile), delimiter=',')
        for row in reader:
            day, cost = int(row[1]), int(row[7])
            tdict[day].append(cost)

        users[tower] = tdict

    tower_lst_ps = [(k,v) for k,v in towers.items()]
    tower_lst_ps.sort(key=lambda x:x[1])
    towers_lst = [p[0] for p in tower_lst_ps]

    rows = list()
    first_row = [-1]
    first_row.extend(towers_lst)
    rows.append(first_row)

    for day in range(1, days + 1):
        daily_lst = [day]
        for tower in towers_lst:
            lst = users[tower].get(day, list())
            cost = 0
            if len(lst) is not 0:
                cost = sum(lst)/len(lst)
            daily_lst.append(cost)
        rows.append(daily_lst)
    mat = np.array(rows)
    np.savetxt(ofile, mat)


def compute_nationality_feature(idir, odir, nat_ptrs, days):
    """
    Generates the counts of each nationality per tower per day
    :param idir: input user directories
    :param odir: the output directory
    :param nat_ptrs: the set of characteristic nationality pointers
    :param days: the number of days in the month
    """
    users = dict()
    for ifile in get_csv_files(idir):
        tower = int(ifile.replace('.csv', ''))
        tdict = defaultdict(list)

        reader = csv.reader(open(idir + '/' + ifile), delimiter=',')
        for row in reader:
            day, nat_in = int(row[1]), row[6]
            nat = nat_ptrs.get(nat_in, None)
            if nat is not None:
                tdict[day].append(nat)

        users[tower] = tdict

    towers = list(users.keys())
    nats = list(set(nat_ptrs.values()))

    first_row = [-1]
    first_row.extend([int(i) for i in nats])

    for day in range(1, days + 1):
        ofile = odir + str(day) + '.csv'
        rows = list()
        rows.append(first_row)
        for tower in towers:
            olst = [int(tower)]
            counts = Counter(users[tower].get(day, list()))
            for nat in nats:
                olst.append(counts.get(nat, 0))
            rows.append(olst)
        omat = np.array(rows)
        np.savetxt(ofile, omat)


def compute_connectivity(idir, odir, tower_ptrs, days):
    """
    Computes the connectivity feature
    :param idir: input directory
    :param odir: output directory to write to
    :param tower_ptrs: the set of characteristic tower pointers
    :param days: the number of days in the month
    """
    users = dict()
    towers, position = dict(), 0
    for ifile in get_csv_files(idir):
        tower = int(ifile.replace('.csv', ''))
        towers[tower] = position # Maps to position in eventual matrix
        position += 1

        tdict = defaultdict(list)

        reader = csv.reader(open(idir + '/' + ifile), delimiter=',')
        for row in reader:
            day, t1 = int(row[1]), row[5]
            ch_tower = tower_ptrs.get(t1, None)
            if ch_tower is not None:
                tdict[day].append(ch_tower)

        users[tower] = tdict

    tower_lst_ps = [(k,v) for k,v in towers.items()]
    tower_lst_ps.sort(key=lambda x:x[1])
    towers_lst = [p[0] for p in tower_lst_ps]

    first_row = [-1]
    first_row.extend([int(i) for i in towers_lst])

    for day in range(1, days + 1):
        ofile = odir + str(day) + '.csv'
        rows = [first_row]

        for tower in towers_lst:
            new_row = [int(tower)]
            counts = Counter(users[tower].get(day, list()))
            new_row.extend([counts.get(str(t), 1) if t != tower else 0 for t in towers_lst])
            rows.append(new_row)
        omat = np.array(rows)
        np.savetxt(ofile, omat)


def generate_features(odir, references, days_in_month=31):
    """
    Runs the generation of the given features and writes to the given out directory
    :param odir: the out directory currently being used to write to
    :param references: a ReferencePointers object with the reference pointers for the towers and countries
    """
    odirs = construct_odir(odir)

    compute_jaccard_feature(odirs.users, odirs.jaccs, days_in_month)
    compute_phonecosts(odirs.users, odirs.costs, days_in_month)
    compute_nationality_feature(odirs.users, odirs.nats, references.char_countries, days_in_month)
    compute_connectivity(odirs.users, odirs.conns, references.char_towers, days_in_month)

if __name__ == "__main__":
    odir = './processed_data'

    tower_file = odir + '/characteristic_towers.csv'
    towers_dict = read_pointers(tower_file)

    country_file = odir + '/characteristic_countries.csv'
    countries_dict = read_pointers(country_file)

    ref_ptrs = ReferencePointers(char_towers=towers_dict,
                                 char_countries=countries_dict)

    generate_features(odir, ref_ptrs)
