import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot.file_reader import read_file
from plot.mapping import color_map, dataset_map, metric_map, marker_map


def plot_knn(result_dir, dataset_name, metric, n_round, algorithms: list, ks: list, save_path=None):
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
        for k in ks:
            scores_per_noise = []
            noise_flag = "_noise_0.2" if dataset_name in ['syn', 'boone', 'frog'] else ""
            for i in range(n_round):
                if algo.lower() in ['solo']:
                    file_name = "../no_priv/{}_A{}_{}.out".format(dataset_name, noise_flag, i)
                elif algo.lower() in ['exact']:
                    file_name = "../no_priv/{}_exact{}_{}.out".format(dataset_name, noise_flag, i)
                else:
                    file_name = "{}_{}_k_{}_{}.out".format(dataset_name, algo_id, k, i)

                file_path = os.path.join(result_dir, file_name)
                scores, time_sec = read_file(file_path, [metric])
                score = scores[0]  # only one metric is allowed
                scores_per_noise.append(score)
            avg_score = np.average(scores_per_noise)
            min_score = np.min(scores_per_noise)
            max_score = np.max(scores_per_noise)

            if algo.lower() in ['featuresim', 'fedsim', 'top1sim', 'avgsim']:
                scores_per_algo.append(avg_score)
                errors_per_algo[0].append(avg_score - min_score)
                errors_per_algo[1].append(max_score - avg_score)
            elif algo.lower() in ['solo', 'exact']:
                scores_per_algo = [avg_score for _ in range(len(ks))]
                errors_per_algo[0] = [avg_score - min_score for _ in range(len(ks))]
                errors_per_algo[1] = [max_score - avg_score for _ in range(len(ks))]
                break
            else:
                assert False

        ax.errorbar(ks, scores_per_algo, marker=marker_map[algo], yerr=errors_per_algo, label=algo,
                    color=color_map[algo], capsize=3)
        print("Algorithm {} printed".format(algo))

    # plt.xscale('log', base=10)
    plt.xlabel(r'Number of neighbors $K$')
    ax.set_ylabel(metric_map[metric])
    ax.set_title(dataset_map[dataset_name])
    fig.set_size_inches(8, 5)
    # lgd = ax.legend()

    fig.tight_layout()

    plt.show()
    if save_path:
        fig.savefig(save_path)

    # fig_leg = plt.figure()
    # ax_leg = fig_leg.add_subplot(111)
    # # add the legend from the previous axes
    # ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=len(algorithms))
    # # hide the axes frame and the x/y labels
    # ax_leg.axis('off')
    #
    # fig_leg.savefig("fig/legend_long.png", bbox_inches='tight', pad_inches=.02)


if __name__ == '__main__':
    plt.rcParams["font.size"] = 20
    os.chdir(sys.path[0] + "/../../")  # change working directory
    # plot_knn(result_dir="./out/performance/beijing/knn", dataset_name="beijing", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[10, 20, 40, 60, 80, 100], save_path="fig/beijing_knn.png")
    # plot_knn(result_dir="./out/performance/hdb/knn", dataset_name="hdb", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[3, 5, 10, 20, 30, 40, 50], save_path="fig/hdb_knn.png")
    # plot_knn(result_dir="./out/performance/game/knn", dataset_name="game", metric="Accuracy", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Exact', 'Solo'],
    #          ks=[3, 5, 10, 20, 30, 40, 50], save_path="fig/game_knn.png")
    # plot_knn(result_dir="./out/performance/song/knn", dataset_name="song", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[3, 5, 10, 20, 30, 40, 50], save_path="fig/song_knn.png")
    # plot_knn(result_dir="./out/performance/ny/knn", dataset_name="ny", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[3, 5, 10, 20, 30, 40, 50], save_path="fig/ny_knn.png")
    # plot_knn(result_dir="./out/performance/boone/knn", dataset_name="boone", metric="Accuracy", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[5, 10, 20, 40, 60, 80, 100], save_path="fig/boone_knn.png")
    # plot_knn(result_dir="./out/performance/frog/knn", dataset_name="frog", metric="Accuracy", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[5, 10, 20, 40, 60, 80, 100], save_path="fig/frog_knn.png")
    # plot_knn(result_dir="./out/performance/syn/knn", dataset_name="syn", metric="Accuracy", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[5, 10, 20, 40, 60, 80, 100], save_path="fig/syn_knn.png")
    plot_knn(result_dir="./out/performance/company/knn", dataset_name="company", metric="R2_Score", n_round=5,
             algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Exact', 'Solo'],
             ks=[5, 10, 20, 30, 40, 50], save_path="fig/company_knn.png")


    # plot_knn(result_dir="./out/beijing/knn", dataset_name="beijing", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[10, 20, 40, 60, 80, 100], save_path="fig/beijing_knn.png")
    # plot_knn(result_dir="./out/hdb/knn", dataset_name="hdb", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[5, 10, 20, 30, 40, 50], save_path="fig/hdb_knn.png")
    # plot_knn(result_dir="./out/game/knn", dataset_name="game", metric="Accuracy", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Exact', 'Solo'],
    #          ks=[3, 5, 10, 20, 30, 40, 50], save_path="fig/game_knn.png")
    # plot_knn(result_dir="./out/song/knn", dataset_name="song", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Exact', 'Solo'],
    #          ks=[5, 20, 30, 40, 50], save_path="fig/song_knn.png")
    # plot_knn(result_dir="./out/ny/knn", dataset_name="ny", metric="R2_Score", n_round=5,
    #          algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #          ks=[5, 10, 20, 30, 40, 50], save_path="fig/ny_knn.png")
