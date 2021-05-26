import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot.file_reader import read_file
from plot.mapping import color_map, dataset_map


def plot_noise(result_dir, dataset_name, metric, n_round, algorithms: list, noises: list, save_path=None, decimal=2):
    assert os.path.isdir(result_dir), "{} is not a directory".format(result_dir)

    fig, ax = plt.subplots()
    x = np.arange(len(noises))
    width = 0.1    # width of bars

    for i_algo, algo in enumerate(algorithms):
        if algo == 'Combine':
            algo_id = 'all'
        elif algo == 'Solo':
            algo_id = 'A'
        else:
            algo_id = algo.lower()

        scores_per_algo = []
        errors_per_algo = [[], []]
        for noise in noises:
            scores_per_noise = []
            for i in range(n_round):
                file_name = "{}_{}_0.0_noise_{:.{prev}f}_{}.out".format(dataset_name, algo_id, noise, i, prev=decimal)
                file_path = os.path.join(result_dir, file_name)
                score, time_sec = read_file(file_path, [metric])
                scores_per_noise.append(score)
            avg_score = np.average(scores_per_noise)
            min_score = np.min(scores_per_noise)
            max_score = np.max(scores_per_noise)

            scores_per_algo.append(avg_score)
            errors_per_algo[0].append(avg_score - min_score)
            errors_per_algo[1].append(max_score - avg_score)

        # plot this algorithm
        if len(algorithms) // 2 == 0:
            pos = x - width * (len(algorithms) // 2) + width / 2 + width * i_algo
        else:
            pos = x - width * (len(algorithms) // 2) + width * i_algo
        ax.bar(pos, scores_per_algo, width, yerr=errors_per_algo, label=algo, capsize=2, color=color_map[algo])

        print("Algorithm {} printed".format(algo))

    ax.set_ylabel(metric)
    ax.set_ylim([0.0, 1.0])
    ax.set_title(dataset_map[dataset_name])
    ax.set_xticks(x)
    ax.set_xticklabels(noises)
    ax.set_xlabel("Noise on identifiers")

    fig.tight_layout()
    lgd = ax.legend(loc='lower right')
    plt.show()

    if save_path:
        fig.savefig(save_path)


if __name__ == '__main__':
    plt.rcParams["font.size"] = 16
    os.chdir(sys.path[0] + "/../../")  # change working directory
    plot_noise(result_dir="./out/syn/no_priv", dataset_name="syn", metric="Accuracy", n_round=5,
               algorithms=['Combine', 'FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'], decimal=1,
               noises=[0.0, 0.1, 0.2], save_path="fig/syn_noise.png")
    plot_noise(result_dir="./out/boone/no_priv", dataset_name="boone", metric="Accuracy", n_round=5,
               algorithms=['Combine', 'FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
               noises=[0.0, 0.1, 0.2], save_path="fig/boone_noise.png", decimal=1)
    plot_noise(result_dir="./out/frog/no_priv", dataset_name="frog", metric="Accuracy", n_round=5,
               algorithms=['Combine', 'FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
               noises=[0.0, 0.1, 0.2], save_path="fig/frog_noise.png", decimal=1)