import pandas as pd
import os
import sys


def load_msd(msd_path):
    print("Loading MSD from {}".format(msd_path))
    msd_df = pd.read_csv(msd_path)
    msd_df.drop(columns=['title'], inplace=True )

    msd_df.info(verbose=True)

    labels = msd_df['label'].to_numpy()
    msd_data = msd_df.drop(columns=['label']).to_numpy()

    return msd_data, labels


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/song")  # change working directory
    fma_df = pd.read_csv("fma_clean.csv")
    msd_df = pd.read_csv("msd_clean.csv")

    merge_df = fma_df.merge(msd_df, how='inner', on='title')

