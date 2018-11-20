import csv
from helper_funcs import get_csv_files, compare_times
from collections import defaultdict


def get_individual_paths(idir, ofile):
    user_dict = defaultdict(set)
    for ifile in get_csv_files(idir):
        tower = int(ifile.replace('.csv',''))
        reader = csv.reader(open(idir + ifile))
        for row in reader:
            user, date = row[0], row[1:5]
            rec = [tower] + date
            user_dict[user].add(';'.join([str(i) for i in rec]))

    writer = csv.writer(open(ofile, 'w'), delimiter=',')
    user_ct, total_stops = 0, 0
    for user, stops in user_dict.items():
        user_ct += 1
        total_stops += len(stops)
        writer.writerow([user] + [stop for stop in stops])

if __name__ == "__main__":
    idir = './processed_data/users/'
    ofile = './processed_data/user_paths.csv'
    get_individual_paths(idir, ofile)