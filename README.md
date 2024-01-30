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
## Quick start to train(search lr, rho and lambda_hero)
```sh
bash auto_train.sh a2c 32 20 1000000 search_all
# $1 is the name of algorithm
# $2 is the set of quantization bit(QAT), using 32 bit means no QAT
# $3 is the test times(seeds number)
# $4 is timestep for training
# $5 is the choice of training, search all lr and rho, search lr , search rho, search lambda_hero
```

[//]: # ()
[//]: # (### Train model from scratch&#40;QAT&#41;)

[//]: # (```sh)

[//]: # (python train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantize 32 -P --rho 0.05 --track)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (### Enjoy model trained with quantization&#40;PTQ&#41;)

[//]: # ()
[//]: # (```sh)

[//]: # (python enjoy.py --algo dqn --env CartPole-v1 -f quantized/8 --exp-id 2)

[//]: # (```)

[//]: # ()
[//]: # (### Post training quantization&#40;PTQ&#41; )

[//]: # (```sh)

[//]: # (python new_ptq.py --algo dqn --env CartPole-v1 --quantized 8 )

[//]: # (```)

[//]: # ()
[//]: # (### Collate model with different quantization bits and build reward diagram)

[//]: # ()
[//]: # (#### PTQ:)

[//]: # (```sh)

[//]: # (python collate_model.py --algo dqn --env CartPole-v1 --device cuda -f quantized --no-render)

[//]: # (```)

#### Post training quantization(PTQ) of all bits(Script)
```sh
ptq_all.sh a2c logs/a2c/CartPole-v1_32bit_lr0.01_rho0.05_lambda1.0_HERO_1 HERO 1 0.05 0.01 1.0
# optional parameter:
#1. algo 
#2. the directory of trained model
#3. optimize choice, like HERO, SAM or base
#4. seeds, from 1 to 4
#5. rho
#6. learning rate
#7. lambda_hero
```
