import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot.file_reader import read_file
from plot.mapping import color_map, dataset_map, metric_map


def plot_priv(result_dir, dataset_name, metric, n_round, algorithms: list, noises: list, save_path=None):
    assert os.path.isdir(result_dir), "{} is not a directory".format(result_dir)

    fig, ax = plt.subplots()

    for algo in algorithms:
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
                if algo.lower() == 'solo':
                    file_name = "{}_A_{}.out".format(dataset_name, i)
                else:
                    b = int(np.ceil(np.abs(np.log10(noise))))
                    a = int(noise / 10 ** (-b))
                    file_name = "{}_{}_p_{}e-{}_{}.out".format(dataset_name, algo_id, a, b, i)

                file_path = os.path.join(result_dir, file_name)
                scores, time_sec = read_file(file_path, [metric])
                score = scores[0]  # only one metric is allowed
                scores_per_noise.append(score)
            avg_score = np.average(scores_per_noise)
            min_score = np.min(scores_per_noise)
            max_score = np.max(scores_per_noise)

            if algo.lower() in ['featuresim', 'fedsim', 'top1sim']:
                scores_per_algo.append(avg_score)
                errors_per_algo[0].append(avg_score - min_score)
                errors_per_algo[1].append(max_score - avg_score)
            elif algo.lower() in ['avgsim', 'solo']:
                scores_per_algo = [avg_score for _ in range(len(noises))]
                errors_per_algo[0] = [avg_score - min_score for _ in range(len(noises))]
                errors_per_algo[1] = [max_score - avg_score for _ in range(len(noises))]
                break
            else:
                assert False

        ax.errorbar(noises, scores_per_algo, marker='s', yerr=errors_per_algo, label=algo,
                    color=color_map[algo], capsize=3)
        print("Algorithm {} printed".format(algo))

    plt.xscale('log', base=10)
    plt.xlabel(r'Bound of success rate $\tau$')
    ax.set_ylabel(metric_map[metric])
    ax.set_title(dataset_map[dataset_name])
    lgd = ax.legend()
    fig.tight_layout()

    plt.show()
    if save_path:
        fig.savefig(save_path)


if __name__ == '__main__':
    plt.rcParams["font.size"] = 16
    os.chdir(sys.path[0] + "/../../")  # change working directory
    # plot_priv(result_dir="./out/beijing/priv", dataset_name="beijing", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5], save_path="fig/beijing_perturb.png")
    # plot_priv(result_dir="./out/hdb/priv", dataset_name="hdb", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5], save_path="fig/hdb_perturb.png")
    # plot_priv(result_dir="./out/game/priv", dataset_name="game", metric="Accuracy", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 2e-3], save_path="fig/game_perturb.png")
    # plot_priv(result_dir="./out/song/priv", dataset_name="song", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 2e-3], save_path="fig/song_perturb.png")
    plot_priv(result_dir="./out/ny/priv", dataset_name="ny", metric="R2_Score", n_round=5,
              algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
              noises=[1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-6], save_path="fig/ny_perturb.png")
