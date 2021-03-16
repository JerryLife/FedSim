import os
import sys

import pandas as pd
import numpy as np
import tabulate

from plot.file_reader import read_file


def create_table(result_dir, dataset_names, metrics, n_rounds, algorithms):
    n_cols = 1 + sum([len(m) for m in metrics])

    table = []
    for algo in algorithms:
        row_cells = []
        for dataset, metric_list in zip(dataset_names, metrics):
            scores_all_round = []
            for i in range(n_rounds):
                file_name = "{}_{}_{}.out".format(dataset.lower(), algo, i)
                file_path = os.path.join(result_dir, file_name)
                try:
                    scores, time = read_file(file_path, metric_list)
                except FileNotFoundError:
                    scores, time = [np.nan, np.nan], np.nan
                scores_all_round.append(list(reversed(scores)))
            mean_score = np.mean(scores_all_round, axis=0)
            std_score = np.std(scores_all_round, axis=0)
            cell_str = "{:.2f}\\textpm {:.2f} & {:.4f}\\textpm {:.4f}".format(
                mean_score[0], std_score[0], mean_score[1], std_score[1])
            row_cells.append(cell_str)

        row_str = algo.capitalize() + "&" + "&".join(row_cells) + "\\\\"
        table.append(row_str)
    table_str = "\n".join(table)
    return table_str


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")  # change working directory
    table_str = create_table(result_dir="./out/beijing",
                             dataset_names=['beijing', 'ny', 'hdb'],
                             metrics=[['R2_Score', 'RMSE'] for _ in range(3)], n_rounds=5,
                             algorithms=reversed(['fedsim', 'ordersim', 'concatsim', 'top1sim', 'avgsim', 'A']))
    print(table_str)