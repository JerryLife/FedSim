import os
import re
import sys
import traceback

import numpy as np


def read_file(file_path, metrics: list):
    """
    Read information from one file, including best performance and training time
    :param metrics: the metrics to be read from the file (e.g. RMSE, Accuracy)
    :param file_path: path of the file
    :return: metric_values: List, time_sec: int; the order of metric values follows the order of metrics
    """
    try:
        time_sec = None
        metric_values = [np.nan for _ in metrics]
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            for line in reversed(f.readlines()):
                if line.isspace():
                    continue
                elif "time" in line:
                    time_sec = int(line.split()[-1])  # get the value of time in seconds (must be integer)
                elif "Best" in line:
                    break
                else:
                    metric, test_value = line.split()[0], float(line.split()[-1])
                    if metric is not None and metric in metrics:
                        metric_values[metrics.index(metric)] = test_value

        # # return a single value if there is only one metric
        # if len(metric_values) == 1:
        #     metric_values = metric_values[0]

        return metric_values, time_sec
    except ValueError:
        print("Failed to read {}".format(file_path))
        raise ValueError


def read_n_params(file_path):
    try:
        n_params = []
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            content = f.read()
            raw_results = re.findall(r"Trainable params:?\s*((?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?)\n", content)
            # if not raw_results:
            #     raw_results = re.findall(r"Trainable params:?\s*(\d+)\n", content)
            results = []
            for s in raw_results:
                if ',' in s:
                    results.append(int(s.replace(',', '')))
                else:
                    results.append(int(s))
            return np.sum(results)
    except ValueError or FileNotFoundError as e:
        print("Failed to read {}".format(file_path))
        raise e
