import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_file(file_path, metrics: list):
    """
    Read information from one file, including best performance and training time
    :param metrics: the metrics to be read from the file (e.g. RMSE, Accuracy)
    :param file_path: path of the file
    :return: metric_values: List, time_sec: int; the order of metric values follows the order of metrics
    """
    assert os.path.isfile(file_path), "{} does not exist".format(file_path)

    time_sec = None
    metric_values = [np.nan for _ in metrics]
    with open(file_path, "r") as f:
        for line in reversed(f.readlines()):
            if "time" in line:
                time_sec = int(line.split()[-1])    # get the value of time in seconds (must be integer)
            elif "Best" in line:
                break
            else:
                metric, test_value = line.split()[0], float(line.split()[-1])
                if metric in metrics:
                    metric_values[metrics.index(metric)] = test_value

    # return a single value if there is only one metric
    if len(metric_values) == 1:
        metric_values = metric_values[0]

    return metric_values, time_sec


def plot_noise(result_dir, dataset_name, metric, n_round, algorithms: list, noises: list, save_path=None):
    assert os.path.isdir(result_dir), "{} is not a directory".format(result_dir)

    fig, ax = plt.subplots()
    x = np.arange(len(noises))
    width = 0.1    # width of bars

    for i_algo, algo in enumerate(algorithms):
        scores_per_algo = []
        errors_per_algo = [[], []]
        for noise in noises:
            scores_per_noise = []
            for i in range(n_round):
                file_name = "{}_{}_noise_{:.2f}_{}.out".format(dataset_name, algo, noise, i)
                file_path = os.path.join(result_dir, file_name)
                score, time_sec = read_file(file_path, [metric])
                scores_per_noise.append(score)
            avg_score = np.average(scores_per_noise)
            min_score = np.min(scores_per_noise)
            max_score = np.max(scores_per_noise)

            scores_per_algo.append(avg_score)
            errors_per_algo[0].append(min_score - avg_score)
            errors_per_algo[1].append(max_score - avg_score)

        # plot this algorithm
        if len(algorithms) // 2 == 0:
            pos = x - width * (len(algorithms) // 2) + width / 2 + width * i_algo
        else:
            pos = x - width * (len(algorithms) // 2) + width * i_algo
        ax.bar(pos, scores_per_algo, width, yerr=errors_per_algo, label=algo)

        print("Algorithm {} printed".format(algo))

    ax.set_ylabel(metric)
    ax.set_title(dataset_name)
    ax.set_xticks(x)
    ax.set_xticklabels(noises)

    fig.tight_layout()
    lgd = ax.legend(loc='lower right')
    plt.show()

    if save_path:
        fig.savefig(save_path)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")  # change working directory
    plot_noise(result_dir="./out/syn/", dataset_name="syn", metric="Accuracy", n_round=5,
               algorithms=['all', 'ordersim', 'mergesim', 'top1sim', 'concatsim', 'avgsim', 'A'],
               noises=[0.0, 0.1, 0.2], save_path="fig/syn_noise.png")
    plot_noise(result_dir="./out/boone/", dataset_name="boone", metric="Accuracy", n_round=1,
               algorithms=['all', 'ordersim', 'mergesim', 'top1sim', 'concatsim', 'avgsim', 'A'],
               noises=[0.0, 0.1, 0.2], save_path="fig/boone_noise.png")