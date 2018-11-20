import csv, os

def parse_time(date):
    """
    Inputs a string value with the date and time
    :param date: a string of the form:
        'yyyy:mm:dd hh:mm:ss'
    :return: a tuple consisting of [dd, hh, mm, ss]
        as integer values
    """
    try:
        spl = date.split(' ')
        lst = spl[0].split('.')
        lst.extend(spl[1].split(':'))
        return [int(i) for i in lst[2:]]
    except ValueError:
        return None

def compare_times_tt(t1, t2):
    return compare_times(t1[1], t2[1])

def compare_times(t1, t2):
    """
    Returns a boolean whether t1 is earlier than t2
    :param t1, t2: two times to compare of form [dd, hh, mm, ss]
    :return: returns true if t1[i] <= t2[i] for all i
    """
    d1, h1, m1, s1 = t1[0:]
    d2, h2, m2, s2 = t2[0:]

    if d2 < d1:
        return -1
    elif d1 == d2:
        if h2 < h1:
            return -1
        elif h2 == h1:
            if m2 < m1:
                return -1
            elif m1 == m2:
                return 0
    return 1


def jaccard_fn(set1, set2):
    """
    Computes the jaccard dissimilarity metric on two sets
    :params set1, set2: two input sets
    :return: 1- |set1 'intersect' set2| / |set1 'union' set2|
    """
    intersect = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union is 0:
        return 1
    return 1- intersect/union


def read_pointers(icsv):
    reader = csv.reader(open(icsv), delimiter=',')
    odict = dict()

    for row in reader:
        ptr, val = row[0], row[1]
        odict[ptr] = val

    return odict


def get_csv_files(idir, suffix=".csv" ):
    fnames = os.listdir(idir)
    return [ifile for ifile in fnames if ifile.endswith(suffix)]

if __name__ == "__main__":
    """
    Testing the comparison and parse functions
    """
    ttime = '2014.06.02 02:23:38'
    t1 = parse_time(ttime)
    assert t1[0] == 2
    assert t1[1] == 2
    assert t1[2] == 23
    assert t1[3] == 38

    sbefore = '2014.06.02 02:23:22'
    safter = '2014.06.02 02:23:51'

    mbefore = '2014.06.02 02:19:38'
    mafter = '2014.06.02 02:27:38'

    hbefore = '2014.06.02 01:23:38'
    hafter = '2014.06.02 05:23:38'

    dbefore = '2014.06.01 02:23:38'
    dafter = '2014.06.05 02:23:38'

    same = '2014.06.02 02:23:38'

    assert not compare_times(t1, parse_time(sbefore))
    assert not compare_times(t1, parse_time(mbefore))
    assert not compare_times(t1, parse_time(hbefore))
    assert not compare_times(t1, parse_time(dbefore))

    assert compare_times(t1, parse_time(safter))
    assert compare_times(t1, parse_time(mafter))
    assert compare_times(t1, parse_time(hafter))
    assert compare_times(t1, parse_time(dafter))
    assert compare_times(t1, parse_time(same))

    """
    Testing the jaccard distance function
    """