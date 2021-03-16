import os
import sys
import pickle

import pandas as pd


def clean_msd(msd_path, out_clean_msd_path):
    msd_titles = []
    msd_data = []
    msd_labels = []
    print("Reformatting msd data")
    with open(msd_path, 'rb') as f:
        msd_data_labels = pickle.load(f)
        for title, datum, label in msd_data_labels:
            title = "".join(title.split())  # remove all whitespaces
            msd_titles.append(title.lower())
            msd_data.append(datum)
            msd_labels.append(label)
    msd_df = pd.DataFrame(msd_data)
    msd_df['title'] = msd_titles
    msd_df['label'] = msd_labels
    print("Saving to {}".format(out_clean_msd_path))
    msd_df.to_csv(out_clean_msd_path, index=False)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/song")  # change working directory
    clean_msd("msd.pkl", "msd_clean.csv")
