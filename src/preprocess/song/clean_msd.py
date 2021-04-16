import os
import sys
import pickle
import re

import pandas as pd


def generate_msd(million_song_dir, out_msd_path):
    pass

def clean_msd(msd_path, out_clean_msd_path):
    msd_titles = []
    msd_data = []
    msd_labels = []
    print("Reformatting msd data")
    with open(msd_path, 'rb') as f:
        msd_data_labels = pickle.load(f)
        for title, datum, label in msd_data_labels:
            # title = "".join(title.split())  # remove all whitespaces
            title = re.sub(r'\W', '', title)
            if len(title) > 0:
                msd_titles.append(title.lower())
                msd_data.append(datum)
                msd_labels.append(label)
    msd_df = pd.DataFrame(msd_data)
    msd_df['title'] = msd_titles
    msd_df['label'] = msd_labels

    # remove duplicate titles
    msd_df.set_index('title', inplace=True)
    msd_df = msd_df[~msd_df.index.duplicated(keep="first")]

    # filter out extreme years
    msd_df = msd_df[msd_df['label'] > 1970]

    print("Saving to {}".format(out_clean_msd_path))
    msd_df.to_csv(out_clean_msd_path)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/song")  # change working directory
    clean_msd("msd.pkl", "msd_clean.csv")
