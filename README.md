# QoRL: Quantization and optimization of Reinforcement Learning (Stable Baselines3)(updating)

Code for QoRL, a framework for evaluating the effects of quantization on reinforcement learning policies across different environments, training algorithms and quantization methods.

## Installation

Please install the following SB3 version because of the modifications in SB3
```sh
cd stable-baselines3 
pip install -e .[docs,tests,extra]
```

## Basic Usage
### Train model from scratch(QAT)
```sh
python train.py --algo ppo --env MountainCarContinuous-v0 --device cuda --optimize-choice base --quantize 32 -P
```


### Enjoy model trained with quantization(QAT)

```sh
python enjoy.py --algo ppo --env MountainCarContinuous-v0 --device cuda --optimize-choice base --quantize 8 -f logs/
```


### Enjoy model trained with quantization(PTQ)

```sh
python enjoy.py --algo ppo --env MountainCarContinuous-v0 -f quantized/8 
```

### Post training quantization(PTQ) 
```sh
python new_ptq.py --algo ppo --env MountainCarContinuous-v0 --quantized 8 
```

### Collate model with different quantization bits and build reward diagram

#### QAT:
```sh
python collate_model.py --algo ppo --env MountainCarContinuous --device cuda --optimize-choice base -f logs/ --no-render
```
#### PTQ:
```sh
python collate_model.py --algo ppo --env MountainCarContinuous --device cuda -f quantized --no-render
```

#### Post training quantization(PTQ) of all bits
```sh
ptq_all.sh dqn CartPole-v1 logs/dqn/CartPole-v1_32_base base
```
