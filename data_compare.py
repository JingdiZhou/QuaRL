import wandb
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

SuggestLR = "SuggestedLR"


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-plot-choice", type=str, default="PTQ", help="the choice to plot PTQ or training data")
    parser.add_argument("--wandb-project-name", type=str, default="", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+",
        help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )
    args = parser.parse_args()
    return args


def main():
    # args = params()
    api = wandb.Api()
    runs_data = {}
    choice = 0
    step = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    # if args.wandb_project_name:
    entity, project = "Qorl", "PTQ_a2c_CartPole-v1"  # args.wandb_project_name

    # if args.wandb_plot_chioce == "train":
    if choice:
        runs = api.runs(entity + '/' + project)
        for run in runs:
            print(run.name)
            # if SuggestLR in run.name:  # using Suggested LR
            #     rho = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[4]))
            #     lr = SuggestLR
            #     optimizer = run.name.split("_")[2]
            # else:
            #     lr = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[3]))
            #     rho = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[4]))
            #     optimizer = run.name.split("_")[2]

            # history = run.scan_history(keys=['train/ep_reward_mean', 'evaluation/mean_reward'], page_size=5000)
            history = run.scan_history(keys=['train/ep_reward_mean'])
            train_reward = [row['train/ep_reward_mean'] for row in history]
            print(train_reward)
            # eval_reward = [row['evaluation/mean_reward'] for row in history]
            # print(train_reward)
            # runs_data[f"{lr}_{rho}_{optimizer}_eval_reward"].append(np.array(eval_reward))
            # runs_data[f"{lr}_{rho}_{optimizer}_train_reward"].append(np.array(train_reward))
            # step = run.step
        # visualization
        # sns.set_theme(style="darkgrid")
        # train_reward_mean
        # train_reward_mean = np.vstack(runs_data[f"{lr}_{rho}_{optimizer}_train_reward"])
        # eval_reward_mean = np.vstack(runs_data[f"{lr}_{rho}_{optimizer}_eval_reward"])
        # df = pd.DataFrame(train_reward_mean).melt(var_name='step',value_name='reward')
        # print(df)
        # sns.lineplot(x='step',y='reward',data=df)
        # plt.show()
    else:

        runs = api.runs(entity + '/' + project)
        num = 0
        sns.set_theme(style="darkgrid")
        for run in runs:
            num += 1
            if num == 1:
                if SuggestLR in run.name:  # using Suggested LR
                    rho = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[5]))
                    lr = SuggestLR
                    optimizer = run.name.split("_")[3]
                else:
                    lr = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[4]))
                    rho = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[5]))
                    optimizer = run.name.split("_")[3]
                runs_data[f"{lr}_{rho}_{optimizer}_ptq_reward"] = []
                history = run.scan_history(keys=['PTQ/reward'])
                ptq_reward = [row['PTQ/reward'] for row in history]
                runs_data[f"{lr}_{rho}_{optimizer}_ptq_reward"].append(np.array(ptq_reward))
                print(1111)
            elif num == 2 or num ==3 or num ==4 or num ==5:
                if SuggestLR in run.name:  # using Suggested LR
                    rho = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[5]))
                    lr = SuggestLR
                    optimizer = run.name.split("_")[3]
                else:
                    lr = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[4]))
                    rho = ''.join(re.findall(r"\d+\.\d+", run.name.split("_")[5]))
                    optimizer = run.name.split("_")[3]
                history = run.scan_history(keys=['PTQ/reward'])
                ptq_reward = [row['PTQ/reward'] for row in history]
                runs_data[f"{lr}_{rho}_{optimizer}_ptq_reward"].append(np.array(ptq_reward))

        # visualization
        for key in runs_data.keys():
            ptq_value = np.mean(np.array(runs_data[key]), axis=0)
            sns.lineplot(ptq_value)
        plt.show()


if __name__ == "__main__":
    main()
