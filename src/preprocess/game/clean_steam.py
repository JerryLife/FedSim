import pandas as pd
import os
import sys


def clean_steam(steam_path, out_steam_path):
    steam_df = pd.read_csv(steam_path, parse_dates=['release_date'])

    steam_df.dropna(inplace=True)
    steam_df.drop(columns=['appid', 'developer', 'publisher', ''])



if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/game")  # change working directory
    clean_steam("steam.csv", "steam_clean.csv")