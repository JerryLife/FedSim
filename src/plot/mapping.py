color_map = {
    'Solo': 'C0',
    'FedSim': 'C3',
    'Top1Sim': 'C1',
    'FeatureSim': 'C2',
    'Combine': 'C4',
    'AvgSim': 'C5',
    'Exact': 'C6'
}

marker_map = {
    'Solo': '^',
    'FedSim': 's',
    'Top1Sim': 'o',
    'FeatureSim': 'v',
    'Combine': 'P',
    'AvgSim': 'x',
    'Exact': 'D'
}

dataset_map = {
    'beijing': 'house',
    'hdb': 'hdb',
    'song': 'song',
    'ny': 'taxi',
    'game': 'game',
    'syn': 'sklearn',
    'frog': 'frog',
    'boone': 'boone',
    'company': 'company'
}

metric_map = {
    'R2_Score': r'$R^2$',
    'Accuracy': 'Accuracy',
    'RMSE': 'RMSE'
}

# 'fedsim', 'mlp', 'disable_sort', 'disable_weight', 'avgsim', 'featuresim', 'top1sim', 'exact', 'A'
algo_map = {
    'fedsim': 'FedSim',
    'mlp': 'FedSim (w/o CNN)',
    'disable_sort': 'FedSim (w/o Sort)',
    'disable_weight': 'FedSim (w/o Weight)',
    'avgsim': 'AvgSim',
    'featuresim': 'FeatureSim',
    'top1sim': 'Top1Sim',
    'exact': 'Exact',
    'A': 'Solo'
}