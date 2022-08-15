import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from plot.file_reader import read_file
from plot.mapping import color_map, dataset_map, metric_map, marker_map


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
        full_dataset_name = dataset_name + "_noise_0.2" if dataset_name in ['syn', 'boone', 'frog'] else dataset_name
        for noise in noises:
            scores_per_noise = []
            for i in range(n_round):
                if algo.lower() == 'solo':
                    file_name = "../no_priv/{}_A_{}.out".format(full_dataset_name, i)
                elif algo.lower() == 'avgsim':
                    file_name = "../no_priv/{}_avgsim_{}.out".format(full_dataset_name, i)
                else:
                    b = -int(np.ceil(np.abs(np.log10(noise))))
                    a = int(noise / 10 ** b)
                    file_name = "{}_{}_p_{}e-{}_{}.out".format(dataset_name, algo_id, a, abs(b), i)

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

        ax.errorbar(noises, scores_per_algo, marker=marker_map[algo], yerr=errors_per_algo, label=algo,
                    color=color_map[algo], capsize=3)
        print("Algorithm {} printed".format(algo))

    plt.xscale('log', base=10)
    plt.xlabel(r'Bound of success rate $\tau$')
    ax.set_ylabel(metric_map[metric])
    ax.set_title(dataset_map[dataset_name])
    # lgd = ax.legend()

    fig.tight_layout()
    fig.set_size_inches(8, 5)

    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=.02)

    # fig_leg = plt.figure()
    # ax_leg = fig_leg.add_subplot(111)
    # # add the legend from the previous axes
    # ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=len(algorithms))
    # # hide the axes frame and the x/y labels
    # ax_leg.axis('off')
    #
    # fig_leg.savefig("fig/legend.png", bbox_inches='tight', pad_inches=.02)


def plot_dp_tau(n, bf_dim ,bf_mean, bf_std, delta=1e-5):
    sigma = np.array([1., 2., 4., 8., 16., 32.])
    noise_to_p = lambda x: erf(np.sqrt(x ** 2 + 1) / (2 * np.sqrt(2) * x * bf_std))
    tau = [noise_to_p(x) for x in sigma]
    sensitivity = n * np.maximum(np.abs((bf_dim + bf_mean) / bf_std), np.abs((-bf_dim + bf_mean) / bf_std))
    # eps = sensitivity * np.sqrt(2 * np.log(1.25/delta)) / sigma
    eps = (sensitivity / sigma) ** 2 / 2

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(sigma, tau, color='C0', marker='^', label=r'$\tau$')
    lns2 = ax2.plot(sigma, eps, color='C1', marker='s', label=r'$\epsilon$')
    ax1.set_xlabel(r'$\sigma$')
    ax1.set_ylabel(r'$\tau$')
    ax2.set_ylabel(r'$\epsilon$')
    fig.tight_layout()
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)
    fig.savefig("fig/dp_tau.jpg")

    pass


if __name__ == '__main__':
    plt.rcParams["font.size"] = 20
    os.chdir(sys.path[0] + "/../../")  # change working directory
    # plot_priv(result_dir="out/performance/beijing/priv", dataset_name="beijing", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 1e-4], save_path="fig/beijing_perturb.png")
    # plot_priv(result_dir="out/performance/hdb/priv", dataset_name="hdb", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 1e-4], save_path="fig/hdb_perturb.png")
    # plot_priv(result_dir="out/performance/game/priv", dataset_name="game", metric="Accuracy", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-1, 1e-2], save_path="fig/game_perturb.png")
    # plot_priv(result_dir="out/performance/song/priv", dataset_name="song", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-1, 1e-2], save_path="fig/song_perturb.png")


    # plot_priv(result_dir="out/performance/ny/priv", dataset_name="ny", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 1e-4], save_path="fig/ny_perturb.png")

    # plot_priv(result_dir="out/performance/boone/priv", dataset_name="boone", metric="Accuracy", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2], save_path="fig/boone_perturb.png")

    # plot_priv(result_dir="out/performance/frog/priv", dataset_name="frog", metric="Accuracy", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2], save_path="fig/frog_perturb.png")

    # plot_priv(result_dir="out/performance/beijing/priv", dataset_name="beijing", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5], save_path="fig/beijing_perturb.png")
    # plot_priv(result_dir="out/performance/hdb/priv", dataset_name="hdb", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'FeatureSim', 'AvgSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5], save_path="fig/hdb_perturb.png")
    # plot_priv(result_dir="out/performance/game/priv", dataset_name="game", metric="Accuracy", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 2e-3], save_path="fig/game_perturb.png")
    # plot_priv(result_dir="out/performance/song/priv", dataset_name="song", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 2e-3], save_path="fig/song_perturb.png")
    # plot_priv(result_dir="out/performance/ny/priv", dataset_name="ny", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-6], save_path="fig/ny_perturb.png")
    # plot_priv(result_dir="out/performance/frog/priv", dataset_name="frog", metric="R2_Score", n_round=5,
    #           algorithms=['FedSim', 'Top1Sim', 'AvgSim', 'FeatureSim', 'Solo'],
    #           noises=[1e-0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3], save_path="fig/frog_perturb.png")

    bf_dim = 1575520 + 1164064
    bf_mean = -46237.78
    bf_std = 21178.86
    n = 141051
    plot_dp_tau(n, 1, bf_mean, bf_std)

