#!/bin/bash
set -e
#########################################
#Positional Parameters
# $1 is the name of algorithm
# $2 is the name of environment
# $3 is the set of quantization bit(QAT), using 32 bit means no QAT
# $4 is the test times(seeds number)
# $5 is timestep for training
# $6 is the choice of training, search all lr and rho, search lr , search rho
# $7 is the choice of doing PTQ to all trained model and visualize them to curves
########################################

Optimizer=("SAM" "base")
Learning_rate=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)
Rho=(0.01 0.02 0.05 0.1 0.2 0.5)
echo "Grid search of Learning rate:[${Learning_rate[*]}]"
echo "Grid search of rho:[${Rho[*]}]"

# 1.Grid search lr and rho simultaneously by default
if [ -z "$6" ]||[ "$6" = "search_all" ];then
  for opt in ${Optimizer[*]};do
    for lr in ${Learning_rate[*]};do
      for rho in ${Rho[*]};do
        for ((i=1;i<=$4;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
          do
          echo "Test learning_rate: $lr, rho: $rho"
          python train.py --algo $1 --env $2 --device cuda --optimize-choice $opt --quantize $3 -P --rho $rho -params learning_rate:$lr --track -n $5
        done
      done
    done
  done

# 2.Grid search rho ,keep using lr suggested by default(rl_baselines3_zoo)
elif [ "$6" = "search_rho" ];then
  for opt in ${Optimizer[*]};do
    for rho in ${Rho[*]};do
        for ((i=1;i<=$4;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
          do
          echo "Test rho: $rho"
          python train.py --algo $1 --env $2 --device cuda --optimize-choice $opt --quantize $3 -P --rho $rho  --track -n $5
        done
      done
    done

# 3.Grid search lr, keep using rho 0.05
elif [ "$6" = "search_lr" ];then
  for opt in ${Optimizer[*]};do
    for lr in ${Learning_rate[*]};do
      for ((i=1;i<=$4;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
        do
        echo "Test learning_rate: $lr"
        python train.py --algo $1 --env $2 --device cuda --optimize-choice $opt --quantize $3 -P --rho 0.05 -params learning_rate:$lr --track -n $5
      done
    done
  done
fi

#
## do PTQ and plot the training data
#if [ -n "$8" ];then
#  dirs=$(ls -F $9 |grep /$)
#  for dir in $dirs;do
#    dir=${dir:0:-1}
#    echo "model path:$9/$dir"
#    for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 6 5 4 3 2 1; do
#      python new_ptq.py --algo $1 --env $2 --quantized $i  -f "$9/$dir" --optimize-choice $3
#    done
#    python plot_PTQ.py --algo $1 --env $2 --device cuda --quantized/ --no-render
#done
##python collate_model.py --algo $1 --env $2 --device cuda  -f quantized/ --no-render --exp-id $5
##python plot_PTQ.py --algo $1 --env $2 --device cuda --quantized/ --no-render
#fi

# auto_train.sh a2c CartPole-v1 32 20 1000000 search_all

