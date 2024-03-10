#!/bin/bash

set -e


#for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 6 5 4 3 2 1; do
#  echo "quantifying model $f"
#  python new_ptq.py --algo $1 --env $2 --quantized $i  -f $3 --optimize-choice $4 --rho $6 --learning_rate $7 --lambda_hero $8
#done
#python collate_model.py --algo $1 --env $2 --device cuda  -f quantized/ --no-render --exp-id $5 --track --rho $6 --learning-rate $7 --lambda_hero $8 --optimize-choice $4

for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 6 5 4 3 2 1; do
  echo "quantifying model $3"
  python new_ptq.py --algo $1 --env $2 --quantized $i  -f $3
done
python collate_model.py --algo $1 --env $2 --device cuda  -f quantized --no-render --exp-id $5 --track --rho $6 --learning-rate $7 --lambda_hero $8 --optimize-choice $4 --wandb-project-name a2c_MountainCarContinuous-v0_PTQ

