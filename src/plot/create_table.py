import os
import sys

import pandas as pd
import numpy as np
import tabulate

from plot.file_reader import read_file


def create_table(result_dir, priv_dir, dataset_names, metrics, n_rounds, algorithms):
    n_cols = 1 + sum([len(m) for m in metrics])

    table = []
    for algo in algorithms:
        row_cells = []
        for dataset, metric_list in zip(dataset_names, metrics):
            scores_all_round = []
            for i in range(n_rounds):
                file_name = "{}_{}_{}.out".format(dataset.lower(), algo, i)
                file_path = os.path.join(result_dir, dataset.lower(), priv_dir, file_name)
                try:
                    scores, time = read_file(file_path, metric_list)
                except (FileNotFoundError, ValueError):
                    scores, time = [np.nan, np.nan], np.nan
                scores_all_round.append(list(reversed(scores)))

            mean_score = np.mean(scores_all_round, axis=0)
            std_score = np.std(scores_all_round, axis=0)
            if len(metric_list) == 2:
                cell_str = "{:.2f}\\textpm {:.2f} & {:.4f}\\textpm {:.4f}".format(
                    mean_score[0], std_score[0], mean_score[1], std_score[1])
            elif len(metric_list) == 1:
                cell_str = "{:.2f}\\textpm {:.2f}\%".format(
                    mean_score[0] * 100, std_score[0] * 100)
            else:
                assert False
            row_cells.append(cell_str)

        row_str = algo.capitalize() + "&" + "&".join(row_cells) + "\\\\"
        table.append(row_str)
    table_str = "\n".join(table)
    return table_str


def create_time_table(result_dir, priv_dir, dataset_names, n_rounds, algorithms):
    table = []
    for algo in algorithms:
        row_cells = []
        for dataset in dataset_names:
            time_all_round = []
            for i in range(n_rounds):
                if dataset.lower() in ['syn', 'boone', 'frog']:
                    file_name = "{}_{}_noise_0.2_{}.out".format(dataset.lower(), algo, i)
                else:
                    file_name = "{}_{}_{}.out".format(dataset.lower(), algo, i)
                file_path = os.path.join(result_dir, dataset.lower(), priv_dir, file_name)
                try:
                    scores, time = read_file(file_path, [])
                except (FileNotFoundError, ValueError):
                    if not (algo.lower() in ['exact'] and dataset.lower() not in ['game', 'song']):
                        print("Failed to read file {}".format(file_path))
                    scores, time = [np.nan, np.nan], np.nan
                assert time is not None, "Failed to read file {}".format(file_path)
                time_all_round.append(time)
            mean_time = np.mean(time_all_round)
            cell_str = "{:.2f}".format(mean_time)
            row_cells.append(cell_str)
        row_str = algo.capitalize() + "&" + "&".join(row_cells) + "\\\\"
        table.append(row_str)
    table_str = "\n".join(table)
    return table_str

if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")  # change working directory
    # table_str = create_table(result_dir="./out/performance", priv_dir="no_priv",
    #                          dataset_names=['beijing', 'ny', 'hdb'],
    #                          metrics=[['R2_Score', 'RMSE'] for _ in range(3)], n_rounds=5,
    #                          algorithms=reversed(['fedsim', 'mlp', 'top1sim', 'avgsim', 'featuresim', 'A']))
    # print(table_str)
    #
    # table_str2 = create_table(result_dir="./out/performance", priv_dir="no_priv",
    #                           dataset_names=['game', 'song'],
    #                           metrics=[['Accuracy'], ['R2_Score', 'RMSE']], n_rounds=5,
    #                           algorithms=reversed(['fedsim', 'mlp', 'top1sim', 'avgsim', 'featuresim', 'A']))
    # print(table_str2)

    table_str = create_table(result_dir="./out/performance", priv_dir="no_priv",
                             dataset_names=['beijing', 'ny', 'hdb', 'game', 'song'],
                             metrics=[['R2_Score', 'RMSE'] for _ in range(3)] + [['Accuracy'], ['R2_Score', 'RMSE']], n_rounds=5,
                             algorithms=reversed(['fedsim', 'mlp', 'top1sim', 'avgsim', 'featuresim', 'exact', 'A']))
    print(table_str)
    time_str = create_time_table(result_dir="./out/performance", priv_dir="no_priv",
                              dataset_names=['syn', 'boone', 'frog', 'beijing', 'ny', 'hdb', 'game', 'song'], n_rounds=5,
                              algorithms=reversed(['fedsim', 'mlp', 'top1sim', 'avgsim', 'featuresim', 'exact']))
    print(time_str)