import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot.file_reader import read_file
from plot.mapping import color_map, dataset_map


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")  # change working directory

    # training time on empty RTX-3090 are directly recorded here
    algorithms = ['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Exact', 'Solo']
    time_record_sec = {
        'sklearn': [2*60+7, 1, 1*60+20, 57, np.nan, 1],
        'boone': [3*60+59, 1, 2*60+45, 2*60+12, np.nan, 1],
        'frog': [10, 1, 6, 4, np.nan, 1],
        'house': [4*60+6, 1, 2*60+35, 1*60+52, np.nan, 1],
        'taxi': [5*60+56, 2, 3*60+22, 2*60+34, np.nan, 1],
        'hdb': [2*60+22, 1, 1*60+40, 34, np.nan, 1],
        'game': [53, 1, 21, 9, 1, 1],
        'song': [8*60+48, 3, 4*60+56, 4*60+3, 2, 2],
    }

    fig, ax = plt.subplots()
    x = np.arange(len(time_record_sec))
    width = 0.1  # width of bars

    for i_algo, algo in enumerate(algorithms):
        if algo == 'Combine':
            algo_id = 'all'
        elif algo == 'Solo':
            algo_id = 'A'
        else:
            algo_id = algo.lower()

        scores_per_algo = np.array(list(time_record_sec.values())).T[i_algo]
        scores_per_algo[np.isnan(scores_per_algo)] = 0

        # plot this algorithm
        if len(algorithms) // 2 == 0:
            pos = x - width * (len(algorithms) // 2) + width / 2 + width * i_algo
        else:
            pos = x - width * (len(algorithms) // 2) + width * i_algo
        ax.bar(pos, scores_per_algo, width, label=algo, capsize=2, color=color_map[algo])

        print("Algorithm {} printed".format(algo))

    ax.set_ylabel("Training time per epoch (sec)")
    # ax.set_ylim([0.0, 1.0])
    # ax.set_title("Training time of different approaches")
    ax.set_xticks(x)
    ax.set_xticklabels(list(time_record_sec.keys()))
    ax.set_xlabel("Datasets")

    fig.set_size_inches(12, 4)
    fig.tight_layout()
    lgd = ax.legend()
    plt.show()

    fig.savefig("fig/training_time.png")
