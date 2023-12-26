#!/bin/bash
set -e

Learning_rate = (0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)
Rho = (0.01 0.02 0.05 0.1 0.2 0.5)
#echo ${Learning_rate[@]}
for lr in ${Learning_rate[*]};do
  for rho in ${Rho[*]};do
    python train.py --algo $1 --env $2 --device cuda --optimize-choice $3 --quantize $4 -P --rho rho -params learning_rate:lr
done
done