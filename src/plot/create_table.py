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


def create_table_solo(result_a_dir, result_b_dir, priv_dir, dataset_names, metrics, n_rounds=5):
    # result of solo A
    row_cells = []
    mean_scores_a = []
    for dataset, metric in zip(dataset_names, metrics):
        scores = []
        for i in range(n_rounds):
            file_name = f"{dataset.lower()}_A_{i}.out"
            file_path = os.path.join(result_a_dir, dataset.lower(), priv_dir, file_name)
            score, _ = read_file(file_path, metric)
            scores.append(score)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        assert len(metric) == 1
        if metric[0] == 'RMSE':
            cell_str = f"{mean_score:.2f}\\textpm {std_score:.2f}"
        elif metric[0] == 'Accuracy':
            cell_str = f"{mean_score * 100:.2f}\\textpm {std_score * 100:.2f}\%"
        else:
            assert False
        row_cells.append(cell_str)
        mean_scores_a.append(mean_score)
    row_str_a = "Primary & " + "&".join(row_cells) + "\\\\"

    # result of solo B
    row_cells = []
    mean_scores_b = []
    for dataset, metric in zip(dataset_names, metrics):
        scores = []
        for i in range(n_rounds):
            file_name = f"{dataset.lower()}_B_{i}.out"
            file_path = os.path.join(result_b_dir, file_name)
            score, _ = read_file(file_path, metric)
            scores.append(score[0])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        assert len(metric) == 1
        if metric[0] == 'RMSE':
            cell_str = f"{mean_score:.2f}\\textpm {std_score:.2f}"
        elif metric[0] == 'Accuracy':
            cell_str = f"{mean_score * 100:.2f}\\textpm {std_score * 100:.2f}\%"
        else:
            assert False
        row_cells.append(cell_str)
        mean_scores_b.append(mean_score)
    row_str_b = "Secondary & " + "&".join(row_cells) + "\\\\"

    # relative difference
    diff = [(a - b) / b for a,b in zip(mean_scores_a, mean_scores_b)]
    diff_str = "&".join([f"{v * 100:.2f}\%" for v in diff])
    row_str_diff = "Difference & " + diff_str

    table_str = "\n".join([row_str_a, row_str_b, row_str_diff])
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

    # table_str = create_table(result_dir="./out/performance", priv_dir="no_priv",
    #                          dataset_names=['beijing', 'ny', 'hdb', 'game', 'song'],
    #                          metrics=[['R2_Score', 'RMSE'] for _ in range(3)] + [['Accuracy'], ['R2_Score', 'RMSE']], n_rounds=5,
    #                          algorithms=reversed(['fedsim', 'mlp', 'disable_sort', 'disable_weight', 'top1sim', 'avgsim', 'featuresim', 'exact', 'A']))
    # print(table_str)
    # time_str = create_time_table(result_dir="./out/performance", priv_dir="no_priv",
    #                           dataset_names=['syn', 'boone', 'frog', 'beijing', 'ny', 'hdb', 'game', 'song'], n_rounds=5,
    #                           algorithms=reversed(['fedsim', 'mlp', 'top1sim', 'avgsim', 'featuresim', 'exact']))
    # print(time_str)

    table_str = create_table_solo(result_a_dir="./out/performance", priv_dir="no_priv", result_b_dir="./out/solo",
                              dataset_names=['beijing', 'ny', 'hdb', 'game', 'song'],
                              metrics=[['RMSE'], ['RMSE'], ['RMSE'], ['Accuracy'], ['RMSE']], n_rounds=5)
    print(table_str)