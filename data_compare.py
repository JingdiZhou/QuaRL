import wandb
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized", help="Quantization bits", default=32, type=int)
    parser.add_argument('--rho', default=0.05, type=float, help='rho of SAM')
    parser.add_argument("--optimize-choice", type=str, default="base", choices=["base", "HERO", "SAM"])
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="",
                        type=str)
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
             "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )
    parser.add_argument("-n", "--n-timesteps",
                        help="Overwrite the number of timesteps, the whole steps of training",
                        default=-1, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1,
                        type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1,
                        type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
             "During hyperparameter optimization n-evaluations is used instead",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--optimization-log-path",
        help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
             "Disabled if no argument is passed.",
        type=str,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=1, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1,
                        type=int)  # checkpoint
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true",
        default=False
    )
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters. "
             "This applies to each optimization runner, not the entire optimization process.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-total-trials",
        help="Number of (potentially pruned) trials for optimizing hyperparameters. "
             "This applies to the entire optimization process and takes precedence over --n-trials if set.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-optimize", "--optimize-hyperparameters", action="store_true", default=False,
        help="Run hyperparameters search"
    )
    parser.add_argument(
        "--no-optim-plots", action="store_true", default=False, help="Disable hyperparameter optimization plots"
    )
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int,
                        default=1)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int,
                        default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization."
             "Default is 1 evaluation per 100k timesteps.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str,
        default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict,
        help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "-conf",
        "--conf-file",
        type=str,
        default=None,
        help="Custom yaml file or python package from which the hyperparameters will be loaded."
             "We expect that python packages contain a dictionary called 'hyperparams' which contains a key for each environment.",
    )
    parser.add_argument("-uuid", "--uuid", action="store_true", default=False,
                        help="Ensure that the run has a unique ID")
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--wandb-project-name", type=str, default="", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+",
        help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )

    args = parser.parse_args()
    return args


def main():
    args = params()
    api = wandb.Api()
    entity, project = "Qorl", "PTQa2c_LunarLander-v2"
    runs = api.runs(entity + '/' + project)
    for run in runs:
        history = run.scan_history()
        # losses = [row for row in history]
        losses = [row['PTQ/reward'] for row in history]
        plt.plot(losses)
        plt.show()


if __name__ == "__main__":
    main()
