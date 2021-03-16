import os
import sys
import pickle

import pandas as pd


def clean_fma(fma_path, out_clean_fma_path):
    fma_titles = []
    fma_data = []
    fma_labels = []
    print("Reformatting fma data")
    with open(fma_path, 'rb') as f:
        fma_data_labels = pickle.load(f)
        for title, datum, label in fma_data_labels:
            title = "".join(title.split())  # remove all whitespaces
            fma_titles.append(title.lower())
            fma_data.append(datum)
            fma_labels.append(label)
    fma_df = pd.DataFrame(fma_data)
    fma_df['title'] = fma_titles
    fma_df['label'] = fma_labels
    print("Saving to {}".format(out_clean_fma_path))
    fma_df.to_csv(out_clean_fma_path, index=False)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/song")  # change working directory
    clean_fma("fma.pkl", "fma_clean.csv")
