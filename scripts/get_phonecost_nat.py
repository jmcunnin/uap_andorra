import os, csv


def process_data():
    dirpath = './processed_data/users/'
    odct = dict()
    for ifile in os.listdir(dirpath):
        reader = csv.reader(open(dirpath + ifile), delimiter=',')
        for row in reader:
            user, nat, ph_cost = row[0], row[-2], row[-1]
            odct[user] = (int(nat), int(ph_cost))

    ofile = './processed_data/user_nat_ph.csv'
    writer = csv.writer(open(ofile, 'w'), delimiter=',')
    for key, values in odct.items():
        row = [key, values[0], values[1]]
        writer.writerow(row)


if __name__ == "__main__":
    process_data()