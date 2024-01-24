# QoRL: Quantization and optimization of Reinforcement Learning (Stable Baselines3)(updating)

Code for QoRL, a framework for evaluating the effects of quantization on reinforcement learning policies across different environments, training algorithms and quantization methods.

## Installation

Please install the following SB3 version because of the modifications in SB3 <br/>
```sh
pip install micronet -i https://pypi.org/simple # QAT module
pip install -r requirements.txt
cd stable-baselines3 # install SB3
pip install -e .
```

## Basic usage

### Optional parameters:
```sh
--rho # rho of SAM optimizer
--quantized # quantization bit setting, 32 bit will be equal to non-quantization
-tb tensorboard_log # using tensorboard
-P # display the progress bar
--no-render # don't render the environment
-params # use the parameters provided by user(not from rl_baseline3_zoo). e.g.-params learning_rate:0.01 buffer_size:256
--track # using wandb to monitor the training
```
## Quick start to train(search lr and rho)
Before training, please delete all the previous trained model in the "logs" that corresponds to the algorithm you want to train
```sh
bash auto_train.sh a2c 32 20 1000000 search_all
# $1 is the name of algorithm
# $2 is the set of quantization bit(QAT), using 32 bit means no QAT
# $3 is the test times(seeds number)
# $4 is timestep for training
# $5 is the choice of training, search all lr and rho, search lr , search rho
```

### Train model from scratch(QAT)
```sh
python train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantize 32 -P --rho 0.05 --track
```


### Enjoy model trained with quantization(QAT)

```sh
python enjoy.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantize 32 -f logs/ --exp-id 2 
#if don't use --exp-id, the latest random seed model will be ran(default)
```


### Enjoy model trained with quantization(PTQ)

```sh
python enjoy.py --algo dqn --env CartPole-v1 -f quantized/8 --exp-id 2
```

### Post training quantization(PTQ) 
```sh
python new_ptq.py --algo dqn --env CartPole-v1 --quantized 8 
```

### Collate model with different quantization bits and build reward diagram

#### QAT:
```sh
python collate_model.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base -f logs/ --no-render
```
#### PTQ:
```sh
python collate_model.py --algo dqn --env CartPole-v1 --device cuda -f quantized --no-render
```

#### Post training quantization(PTQ) of all bits(Script)
```sh
ptq_all.sh dqn CartPole-v1 logs/dqn/CartPole-v1_32bit_base_2 base 2
# the third parameter is the model path, the fourth parameter is the optimize_choice, the fifth parameter is exp-id:random seed
```
### Plot training curves
given a path of a specific directory which contains trained models of different random seeds, for example, logs/dqn, the script will get all the files satisfied the request
to plot all the training data to one curve <br/>
currently support data of SAM and base(HERO later) 
```sh
python plot_outcome.py --algo dqn --env CartPole-v1 -s -f logs/dqn  #-f: the directory path containing the log file(s); -s save figure
```
## Current contributions

|  RL Algo | PTQ                | QAT                |
|----------|--------------------|--------------------|
| ARS      | :heavy_check_mark: |                    | 
| A2C      | :heavy_check_mark: | :heavy_check_mark: | 
| PPO      | :heavy_check_mark: | :heavy_check_mark: | 
| DQN      | :heavy_check_mark: | :heavy_check_mark: | 
| QR-DQN   | :heavy_check_mark: |                    | 
| DDPG     | :heavy_check_mark: |                    | 
| SAC      | :heavy_check_mark: |                    | 
| TD3      | :heavy_check_mark: |                    | 
| TQC      | :heavy_check_mark: |                    | 
| TRPO     | :heavy_check_mark: |                    | 

