import csv, os
from collections import defaultdict, namedtuple
from helper_funcs import parse_time

#################
## Variable Names:
USER_ID = "DS_CDNUMORIGEN"
START_TIME = "DT_CDDATAINICI"
SENDER_TOWER = "ID_CELLA_INI"
RECEIVER_TOWER = "ID_CELLA_FI"
USER_ORIGIN = "ID_CDOPERADORORIGEN"
TAC_CODE = "TAC_IMEI"

DAYS_IN_MONTH = -1

DataIndices = namedtuple('DataIndices', 'user time s_tower r_tower nationality tac_code')
InFiles = namedtuple('InFiles', 'records towers tac_codes country_codes')

##################
# Helper Functions
def get_towers(towersf):
    """
    A function to import the towers and latitude/longitude pairs
    and combines towers in the same position
    :param towersf: An input file with tower positions with three rows:
        [tower_id, latitude, longitude]
    :return: Returns two objects:
        -- A dictionary mapping a tower_id to the representative tower
    """
    groups = defaultdict(list)
    reader = csv.reader(open(towersf), delimiter=',')
    next(reader, None)

    for row in reader:
        tow, lat, long = row[0], row[1], row[2]
        position = (lat, long)
        groups[position].append(tow)

    s = set()
    for k, v in groups.items():
        s.add(v[0])

    chartowers = dict()
    for k, v in groups.items():
        characteristic = v[0]
        for t in v:
            chartowers[t] = characteristic

    return chartowers


def get_indices(index_row):
    """
    Returns the indices of each field as a DataIndices object
    :param index_row: the header row in the input values
    :return: a DataIndices object
    :exception AssertionError: raised when one of the indices is not found.
        Caused by a mispelled header row relative to the variable names defined above
    """
    indices = DataIndices(user=index_row.index(USER_ID),
                          time=index_row.index(START_TIME),
                          s_tower=index_row.index(SENDER_TOWER),
                          r_tower=index_row.index(RECEIVER_TOWER),
                          nationality=index_row.index(USER_ORIGIN),
                          tac_code=index_row.index(TAC_CODE))

    for idx in indices:
        assert idx > -1

    return indices


def get_countrycodes(countriesf):
    """
    Gets a dictionary of country pointers
    :param countriesf: The input file of countries of the form [country, code]
    :return: a dictionary of code -> code pointers where codes from the same country point to one value
    """
    reader = csv.reader(open(countriesf), delimiter=',')
    next(reader, None)

    country_code_ptrs = defaultdict(list)
    for row in reader:
        country_code_ptrs[row[0]].append(int(row[1]))

    code_ptrs = dict()
    for k, v in country_code_ptrs.items():
        characteristic = v[0]
        for val in v:
            code_ptrs[val] = characteristic
    return code_ptrs


def get_tac_codes(tacf):
    reader = csv.reader(open(tacf), delimiter=',')
    out_tacdb = dict()

    for row in reader:
        tac, cost = row[0], int(row[1])
        out_tacdb[tac] = cost

    return out_tacdb



##################
# Main Function
def parse_data_input(ifiles, odir):
    """
    Handles the parsing of the input data file
    :param ifiles: the data file to be parsed
    :param tower_pointers: a dictionary of pointers to characteristic towers
    :param odir: the directory to be written to. Assumes no directory exists and creates the directory
                with an additional '/users/' subdirectory
    :returns: processes the input data and groups by tower returning a list of:
        [user, day, hour, minute, second, receiving_tower, nation of origin, phone_cost]
    """
    users_dir = odir + '/users/'
    if os.path.exists(odir):
        raise ValueError("Out directory exists")
    else:
        os.makedirs(users_dir)

    characteristic_towers = get_towers(ifiles.towers)
    country_ptrs = get_countrycodes(ifiles.country_codes)
    tac_costs = get_tac_codes(ifiles.tac_codes)

    reader = csv.reader(open(ifiles.records), delimiter=';')
    headers = next(reader, None)
    idxs = get_indices(headers)

    users = defaultdict(list)
    print("About to begin processing data")

    for row in reader:
        user, time, send_t = row[idxs.user], row[idxs.time], row[idxs.s_tower]
        rec_t, nat_t, tac_code = row[idxs.r_tower], row[idxs.nationality], row[idxs.tac_code]

        send = characteristic_towers.get(send_t, None)
        if send is None:
            continue

        tparsed = parse_time(time)
        if tparsed is None:
            continue
        else:
            urec = [user] + tparsed
            nat = ''
            if nat_t is not '':
                nat = int(nat_t)
            other_features = [characteristic_towers.get(rec_t, None), country_ptrs.get(nat, None), tac_costs.get(tac_code, None)]
            urec.extend([i if i is not None else -1 for i in other_features])
            users[send].append(urec)

    for k,v in users.items():
        ofile = users_dir + str(k) + ".csv"
        writer = csv.writer(open(ofile, "w"), delimiter=",")
        for row in v:
            writer.writerow(row)

    tower_ofile = odir + 'characteristic_towers.csv'
    writer = csv.writer(open(tower_ofile, 'w'), delimiter=',')
    for k,v in characteristic_towers.items():
        writer.writerow([k,v])

    country_ofile = odir + 'characteristic_countries.csv'
    writer = csv.writer(open(country_ofile, 'w'), delimiter=',')
    for k,v in country_ptrs.items():
        writer.writerow([k,v])

if __name__ == "__main__":
    recordsf = './DWFET_CDR_CELLID_201406.csv'
    towersf = './towers.csv'
    tac_db = './tac_db_prices.csv'
    countriesf = './countrycode.csv'
    out_directory = './processed_data/'
    # Replace above with sys.argv[i] when ready
    ifiles = InFiles(records=recordsf,
                     towers=towersf,
                     tac_codes=tac_db,
                     country_codes=countriesf)
    parse_data_input(ifiles, out_directory)
