import argparse
import csv
import h5py
import pickle
import itertools
import string
import os
import re

parser = argparse.ArgumentParser()
available_datasets = ['fma', 'dma', 'all']
parser.add_argument('-ds', '--dataset', default='all', type=str,
                    help="Aailable datasets: {}".format("|".join(available_datasets)))
parser.add_argument('-f', '--force', action='store_true', help="force re-process dataset")

args = parser.parse_args()
# target of preprocessing:
#     * a list of ([features], year) tags
#     * combined dataset with a large ([features], year) tags

output_dir = 'preprocessed_data'
try:
    os.makedirs(output_dir)
except FileExistsError:
    pass

# read fma
if args.dataset == 'fma' or args.dataset == 'all':
    if os.path.isfile(output_dir + '/fma_pps.pkl') and not args.force:
        print("Loading preprocessed FMA...")
        with open(output_dir + '/fma_pps.pkl', 'rb') as f:
            tracks_fma = pickle.load(f)
    else:
        tracks_fma = {}
        # extract id, title and year
        with open('data/fma_metadata/tracks.csv', 'r') as f:
            headers = []
            # construct header
            headers_reader = csv.reader(f)
            for row in itertools.islice(headers_reader, 1):
                for col in row:
                    headers.append(col)
            for row in itertools.islice(headers_reader, 2):
                for i, col in enumerate(row):
                    headers[i] += "_{}".format(col)

            # read csv file
            tracks_reader = csv.DictReader(f, headers)
            for i, row in enumerate(itertools.islice(tracks_reader, 4, None)):
                # skip the tracks without year tag
                if len(row['album_date_released_']) > 0:
                    tracks_fma[int(row['__track_id'])] = [
                        # TODO: make year a int
                        row['track_title_'], None, int(row['album_date_released_'].split('-')[0])]

                if i % 100 == 0:
                    print('\rFMA (1/2) {}/???'.format(i), end="")

        # extract features
        with open('data/fma_metadata/features.csv', 'r') as f:
            features_reader = csv.reader(f)
            for i, row in enumerate(itertools.islice(features_reader, 4, None)):
                if int(row[0]) in tracks_fma:
                    tracks_fma[int(row[0])][1] = [float(feature)
                                                  for feature in row[1:]]

                if i % 100 == 0:
                    print('\rFMA (2/2) {}/???'.format(i), end="")

        print("\nParsed {} entires from FMA.".format(len(tracks_fma.keys())))
        with open(output_dir + '/fma_pps.pkl', 'wb') as f:
            pickle.dump(tracks_fma, f)
        with open(output_dir + '/fma.pkl', 'wb') as f:
            pickle.dump(list(tracks_fma.values()), f)

# read msd
if args.dataset == 'msd' or args.dataset == 'all':
    if os.path.isfile(output_dir + '/msd_pps.pkl') and not args.force:
        print("Loading preprocessed MSD...")
        with open(output_dir + '/msd_pps.pkl', 'rb') as f:
            tracks_msd = pickle.load(f)
    else:
        tracks_msd = {}
        # extract id, title and year
        with open('data/tracks_per_year.txt', 'r') as f:
            for i, row in enumerate(f):
                row_content = row.split('<SEP>')
                tracks_msd[bytes(row_content[1], encoding='ascii')] = [
                    row_content[3], None, int(row_content[0].strip())]

                if i % 100 == 0:
                    print('\rMSD (1/3) {}/???'.format(i), end="")

        # extract features
        id_idx_map = {}
        with h5py.File('data/msd_summary_file.h5', 'r', rdcc_nbytes=50*1024*1024) as f:
            for i, row in enumerate(f['analysis']['songs']):
                id_idx_map[row['track_id']] = i
                if i % 100 == 0:
                    print('\rMSD (2/3) {}/1000000'.format(i), end="")

            for i, key in enumerate(tracks_msd.keys()):
                if key in id_idx_map:
                    tracks_msd[key][1] = list(
                        f['analysis']['songs'][id_idx_map[key]])[2:-1]
                else:
                    tracks_msd.pop(key)

                if i % 100 == 0:
                    print('\rMSD (3/3) {}/{}'.format(i,
                                                     len(tracks_msd.keys())), end="")

        print("\nParsed {} entires from MSD.".format(len(tracks_msd.keys())))
        with open(output_dir + '/msd_pps.pkl', 'wb') as f:
            pickle.dump(tracks_msd, f)
        with open(output_dir + '/msd.pkl', 'wb') as f:
            pickle.dump(list(tracks_msd.values()), f)

if args.dataset == 'all':
    tracks_common = []
    tracks_fma_aligned = []
    tracks_msd_aligned = []
    tracks_union = []

    regularizer = re.compile('[\W]')
    name_id_map_fma = {regularizer.sub('', value[0]).lower() + str(value[2]): key
            for key, value in tracks_fma.items()}
    name_id_map_msd = {regularizer.sub('', value[0]).lower() + str(value[2]): key
            for key, value in tracks_msd.items()}
    print(len(name_id_map_fma), len(name_id_map_msd))

    common_set = set(name_id_map_fma.keys()).intersection(
        set(name_id_map_msd.keys()))
    # remove misaligned items
#    to_remove=set()
#    for name in common_set:
#        if tracks_fma[name_id_map_fma[name]][2] != tracks_msd[name_id_map_msd[name]][2]:
#            to_remove.add(name)
#    common_set = common_set.difference(to_remove)

    # inner jion
    for i, name in enumerate(common_set):
        tracks_common.append([
            name,
            tracks_fma[name_id_map_fma[name]][1],
            tracks_msd[name_id_map_msd[name]][1],
            tracks_msd[name_id_map_msd[name]][2]])
        tracks_fma_aligned.append(tracks_fma[name_id_map_fma[name]])
        tracks_msd_aligned.append(tracks_msd[name_id_map_msd[name]])

        if i % 100 == 0:
            print('\rAligned (1/2) {}/{}'.format(i, len(common_set)), end="")

    print("\nParsed {} entires in common.".format(len(common_set)))

    union_set = set(name_id_map_fma.keys()).union(set(name_id_map_msd.keys()))
    # remove misaligned
#    to_remove=set()
#    for name in union_set:
#        if (name in name_id_map_fma) and (name in name_id_map_msd) and (tracks_fma[name_id_map_fma[name]][2] != tracks_msd[name_id_map_msd[name]][2]):
#            to_remove.add(name)
#    union_set = union_set.difference(to_remove)

    fma_features_length = len(list(tracks_fma.values())[0][1])
    msd_features_length = len(list(tracks_msd.values())[0][1])
    for i, name in enumerate(union_set):
        tracks_union.append([
            name,
            tracks_fma[name_id_map_fma[name]][1] if name in name_id_map_fma else [0.0] * fma_features_length,
            tracks_msd[name_id_map_msd[name]][1] if name in name_id_map_msd else [0.0] * msd_features_length,
            tracks_msd[name_id_map_msd[name]][2] if name in name_id_map_msd else tracks_fma[name_id_map_fma[name]][2]
        ])

        if i % 100 == 0:
            print('\rAligned (2/2) {}/{}'.format(i, len(union_set)), end="")

    print("\nParsed {} entires in union.".format(len(union_set)))

    with open(output_dir + '/fma_aligned.pkl', 'wb') as f:
        pickle.dump(tracks_fma_aligned, f)

    with open(output_dir + '/msd_aligned.pkl', 'wb') as f:
        pickle.dump(tracks_msd_aligned, f)

    with open(output_dir + '/both.pkl', 'wb') as f:
        pickle.dump(tracks_common, f)

    with open(output_dir + '/union.pkl', 'wb') as f:
        pickle.dump(tracks_union, f)

