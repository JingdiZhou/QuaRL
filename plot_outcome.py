"""
Plot training reward/success rate using EMA and resample(from OpenAI baseline:https://github.com/openai/baselines)
"""
import argparse
import os
import json
from typing import Tuple, List

import pandas as pd
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from pandas import DataFrame
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func

# Activate seaborn
seaborn.set()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
smooth_step = 1.0
resample = 9192


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0  # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _, ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys


def get_monitor_files(path):
    file_list = os.listdir(path)
    return file_list


def load_results(path: str) -> tuple[list[DataFrame], list[DataFrame]]:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """

    monitor_files = get_monitor_files(path)
    print("monitor_files:", monitor_files)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *monitor.csv found in {path}")
    data_frames_SAM, data_frame_base, headers = [], [], []
    for file_name in monitor_files:
        print("###", file_name)
        file_name = os.path.join(path, file_name, "0.monitor.csv")
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            file = os.path.normpath(file_name).split(os.path.sep)
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pd.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frame.sort_values("t", inplace=True)
        data_frame.reset_index(inplace=True)
        data_frame["t"] -= min(header["t_start"] for header in headers)
        if "SAM" in file[-2].split('_')[-2]:
            print("SAM")
            data_frames_SAM.append(data_frame)
        elif "base" in file[-2].split('_')[-2]:
            print("base")
            data_frame_base.append(data_frame)
    return data_frames_SAM, data_frame_base


def plot_train():
    parser = argparse.ArgumentParser("Gather results, plot training reward/success")
    parser.add_argument("-a", "--algo", help="Algorithm to include", type=str, required=True)
    parser.add_argument("-e", "--env", help="Environment(s) to include", type=str, required=True)
    parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
    parser.add_argument("-s", "--savefig", help="save figure(plot) or not", required=True, default=False,
                        action='store_true')
    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int,
                        default=[6.4, 4.8])
    parser.add_argument("--fontsize", help="Font size", type=int, default=14)
    parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
    parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str,
                        default="steps")
    parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["success", "reward", "length"], type=str,
                        default="reward")
    parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=100)

    args = parser.parse_args()

    algo = args.algo
    envs = args.env
    log_path = args.exp_folder
    x_axis = {
        "steps": X_TIMESTEPS,
        "episodes": X_EPISODES,
        "time": X_WALLTIME,
    }[args.x_axis]
    x_label = {
        "steps": "Timesteps",
        "episodes": "Episodes",
        "time": "Walltime (in hours)",
    }[args.x_axis]

    y_axis = {
        "success": "is_success",
        "reward": "r",
        "length": "l",
    }[args.y_axis]
    y_label = {
        "success": "Training Success Rate",
        "reward": "Training Episodic Reward",
        "length": "Training Episode Length",
    }[args.y_axis]

    dirs = []
    num = 0
    dirs.append(log_path)
    Y = []
    Y_mean_SAM, Y_mean_base, Y_std_SAM, Y_std_base = 0, 0, 0, 0
    data_frames_SAM, data_frames_base = load_results(log_path)
    data = [data_frames_SAM, data_frames_base]
    for data_frames in data:
        num += 1
        if not data_frames:  # if no dataframes,skip this loop
            continue
        for data_frame in data_frames:
            if args.max_timesteps is not None:
                data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
            try:
                x, y = ts2xy(data_frame, x_axis)
            except KeyError:
                print(f"No data available for {log_path}")
                continue
            low, high = x[0], x[-1]
            usex = np.linspace(low, high, resample)  # resample can be set
            print(">>>>", low, high)
            Y.append(symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
        if num == 2:
            Y_mean_base = np.mean(Y, axis=0)  # get average values of different random seeds(model)
            Y_std_base = np.std(Y, axis=0)  # get std values of different random seeds(model)
        elif num == 1:
            Y_mean_SAM = np.mean(Y, axis=0)
            Y_std_SAM = np.std(Y, axis=0)
    plt.figure(y_label, figsize=args.figsize)
    plt.title(y_label, fontsize=args.fontsize)
    plt.xlabel(f"{x_label}", fontsize=args.fontsize)
    plt.ylabel(y_label, fontsize=args.fontsize)
    if not isinstance(Y_mean_SAM,int):
        plt.plot(usex, Y_mean_SAM, label="SAM", color='#b36ff6')
        plt.fill_between(usex, Y_mean_SAM - Y_std_SAM, Y_mean_SAM + Y_std_SAM, color='#c875c4', alpha=.4)
    elif not isinstance(Y_mean_base,int):
        plt.plot(usex, Y_mean_base, label="vanilla")
        plt.fill_between(usex, Y_mean_base - Y_std_base, Y_mean_base + Y_std_base, alpha=.4)
    else:
        raise ValueError(f"no valid data to plot, maybe no valid file under the given file path")
    plt.legend()
    if args.savefig:
        if not os.path.exists('pngs/plot_train'):
            os.makedirs('pngs/plot_train')
        plt.savefig(os.path.join('pngs/plot_train', f'training_curve_{args.algo}_{args.env}.png'))
    plt.show()


if __name__ == "__main__":
    plot_train()
