# Get data from wandb

import wandb
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

SuggestedLR = "SuggestedLR"

learning_rate = [0.0001, 0.0005, 0.001, 0.007, 0.005, 0.01, 0.05, 0.1, 0.5, "SuggestedLR"]
Rho = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
optimizer = ["SAM", "base", "HERO"]
combination = [f"lr{lr}_rho{rho}_{opt}" for lr in learning_rate for rho in Rho for opt in optimizer]

data_all = dict.fromkeys(combination)
for key in data_all:
    data_all[key] = []


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-plot-choice", type=str, default="PTQ", help="the choice to plot PTQ or training data")
    parser.add_argument("--wandb-team-name", type=str, default="", help="team name of wandb")
    parser.add_argument("--wandb-project-name", type=str, default="", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+",
        help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )
    args = parser.parse_args()
    return args


def main():
    args = params()
    api = wandb.Api()
    entity, project = args.wandb_team_name, args.wandb_project_name
    algo = args.wandb_project_name.split('_')[1] if args.wandb_plot_choice == "PTQ" else \
        args.wandb_project_name.split('_')[0]
    env = args.wandb_project_name.split('_')[2] if args.wandb_plot_choice == "PTQ" else \
        args.wandb_project_name.split('_')[1]

    if args.wandb_plot_choice == "train":
        runs = api.runs(entity + '/' + project)
        for run in runs:
            # Get related info
            run_name = run.name if args.wandb_plot_choice == "train" else run.name.replace('PTQ_', '')
            rho = ''.join(re.findall(r"\d+\.\d+", run_name.split("_")[4]))
            optimizer = run_name.split("_")[2]
            if SuggestedLR in run_name:
                lr = SuggestedLR
            else:
                lr = ''.join(re.findall(r"\d+\.\d+", run_name.split("_")[3]))

            history = run.scan_history(keys=['evaluation/mean_reward'],page_size=100)  # page_size=5000)
            eval_reward = [row['evaluation/mean_reward'] for row in history]
            print(len(eval_reward))

    elif args.wandb_plot_choice == "system":
        hist_data = []
        x = [1, 2, 3]
        runs = api.runs(entity + '/' + project)
        for run in runs:
            # get related info
            run_name = run.name if args.wandb_plot_choice == "train" else run.name.replace('PTQ_', '')
            rho = ''.join(re.findall(r"\d+\.\d+", run_name.split("_")[4]))
            optimizer = run_name.split("_")[2]
            if SuggestedLR in run_name:
                lr = SuggestedLR
            else:
                lr = ''.join(re.findall(r"\d+\.\d+", run_name.split("_")[3]))
            time = run.summary["_wandb"].runtime
            data_all[f'lr{lr}_rho{rho}_{optimizer}'].append(time)
        for key in data_all.keys():
            if len(data_all[f"{key}"]) != 0:
                data_all[f'{key}'] = np.mean(data_all[f'{key}']) / 60
                print(f"{key}: ", data_all[f'{key}'])
                hist_data.append(data_all[f'{key}'])
        plt.bar(x, hist_data, tick_label=['SAM', 'base', 'HERO'], facecolor='#9999ff', edgecolor='red', width=0.4)
        plt.title(f"time_{algo}_{env}")
        plt.xlabel("Optimizer")
        plt.ylabel('Time(minute)')
        plt.show()

    elif args.wandb_plot_choice == "ptq":
        plt.style.use('ggplot')
        step = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        runs = api.runs(entity + '/' + project)
        for run in runs:
            # get related info
            run_name = run.name if args.wandb_plot_choice == "train" else run.name.replace('PTQ_', '')
            rho = ''.join(re.findall(r"\d+\.\d+", run_name.split("_")[4]))
            optimizer = run_name.split("_")[2]
            if SuggestedLR in run_name:
                lr = SuggestedLR
            else:
                lr = ''.join(re.findall(r"\d+\.\d+", run_name.split("_")[3]))
            history = run.scan_history(keys=['PTQ/reward'])
            ptq = [row['PTQ/reward'] for row in history]
            data_all[f'lr{lr}_rho{rho}_{optimizer}'].append(ptq)

        for key in data_all.keys():
            if len(data_all[f"{key}"]) != 0:
                data_mean = np.mean(data_all[f'{key}'], axis=0)
                data_std = np.std(data_all[f'{key}'], axis=0)
                plt.plot(step, data_mean, label=key)
                plt.fill_between(x=step, y1=data_mean - data_std, y2=data_mean + data_std, alpha=0.3)
        plt.title(f"PTQ_{algo}_{env}")
        plt.xlabel("bit")
        plt.ylabel('Average reward')
        plt.legend()
        plt.savefig(f"PTQ_{algo}_{env}.png")
        plt.show()


if __name__ == "__main__":
    main()
