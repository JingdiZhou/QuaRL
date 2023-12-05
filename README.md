# QoRL:Quantization and optimization of Reinforcement Learning (Stable Baselines3 PyTorch)(updating)


### Train model from scratch(QAT)
``
python train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantize 32 -P
``

### Enjoy model trained with quantization(QAT)

``
python enjoy.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantize 8 -f logs/
``
### Enjoy model trained with quantization(PTQ)

``
python enjoy.py --algo dqn --env CartPole-v1 -f quantized/8 
``
### Post training quantization(PTQ) 
``
python new_ptq.py --algo dqn --env CartPole-v1 --quantized 8 --exp-id 1
``

### Collate model with different quantization bits and build reward diagram

#### QAT:
``
python collate_model.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base -f logs/ --no-render
``
#### PTQ:
``
python collate_model.py --algo dqn --env CartPole-v1 --device cuda -f quantized --no-render
``

#### Post training quantization(PTQ) of all bits
