# QoRL: Quantization and optimization of Reinforcement Learning (Stable Baselines3)(updating)

Code for QoRL, a framework for evaluating the effects of quantization on reinforcement learning policies across different environments, training algorithms and quantization methods.

## Installation

Please install the following SB3 version because of the modifications in SB3 <br/>
About the installation of other packages please refer to rl_baselines_zoo : [rl_baseline3_zoo](https://github.com/DLR-RM/rl-baselines3-zoo#installation)
```sh
pip install micronet -i https://pypi.org/simple # QAT module
cd stable-baselines3 # install SB3
pip install -e .[docs,tests,extra]
```

## Basic usage

### Optional parameters:
```sh
-tb terboard_log # using tensorboard
-P # display the progress bar
--no-render # don't render the environment
```
### Train model from scratch(QAT)
```sh
python train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantize 32 -P
```


### Enjoy model trained with quantization(QAT)

```sh
python enjoy.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantize 8 -f logs/
```


### Enjoy model trained with quantization(PTQ)

```sh
python enjoy.py --algo dqn --env CartPole-v1 -f quantized/8 
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
#### PTQ:
```sh
python collate_model.py --algo ppo --env MountainCarContinuous --device cuda -f quantized --no-render
```

#### Post training quantization(PTQ) of all bits(Script)
```sh
ptq_all.sh dqn CartPole-v1 logs/dqn/CartPole-v1_32_base base
```

## Framework

