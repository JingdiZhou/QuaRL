#!/bin/bash
set -e
#########################################
#Positional Parameters
# $1 is the name of algorithm
# $2 is the set of quantization bit(QAT), using 32 bit means no QAT
# $3 is the test times(seeds number)
# $4 is timestep for training
# $5 is the choice of training, search all lr and rho, search lr , search rho
########################################

#env_dqn=('CartPole-v1' 'MountainCar-v0' 'LunarLander-v2' 'Acrobot-v1')
#env_a2c=('CartPole-v1' 'LunarLander-v2' 'MountainCar-v0' 'Acrobot-v1' 'Pendulum-v1' 'LunarLanderContinuous-v2')
env_a2c=('CartPole-v1' 'LunarLander-v2' 'Pendulum-v1')
#env_sac=('MountainCarContinuous-v0' 'Pendulum-v1' 'LunarLanderContinuous-v2')
Optimizer=("SAM" "base")
#Learning_rate=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)
#Rho=(0.01 0.02 0.05 0.1 0.2 0.5)
Learning_rate=(0.01 0.05)
Rho=(0.01 0.05)
echo "Grid search of Learning rate:[${Learning_rate[*]}]"
echo "Grid search of rho:[${Rho[*]}]"

# 1.Grid search lr and rho simultaneously by default
if [ -z "$5" ]||[ "$5" = "search_all" ];then
  if [ "$1" = "dqn" ];then
  for env in ${env_dqn[*]};do
    for opt in ${Optimizer[*]};do
      for lr in ${Learning_rate[*]};do
        for rho in ${Rho[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr, rho: $rho"
            python train.py --algo $1 --env $env  --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho -params learning_rate:$lr --track -n $4
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i $rho $lr $opt
          done
        done
      done
    done
  done
  elif [ "$1" = "a2c" ];then
    for env in ${env_a2c[*]};do
    for opt in ${Optimizer[*]};do
      for lr in ${Learning_rate[*]};do
        for rho in ${Rho[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr, rho: $rho"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho -params learning_rate:$lr --track -n $4
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i $rho $lr $opt
          done
        done
      done
    done
  done
  elif [ "$1" = "sac" ]; then
    for env in ${env_sac[*]};do
    for opt in ${Optimizer[*]};do
      for lr in ${Learning_rate[*]};do
        for rho in ${Rho[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr, rho: $rho"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho -params learning_rate:$lr --track -n $4
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i $rho $lr $opt
          done
        done
      done
    done
  done
  fi


# 2.Grid search rho, keep using lr suggested by default(rl_baselines3_zoo)
elif [ "$5" = "search_rho" ];then
  if [ "$1" = "dqn" ];then
    for env in ${env_dqn[*]};do
      for opt in ${Optimizer[*]};do
        for rho in ${Rho[*]};do
            for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
              do
              echo "Test rho: $rho"
              python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho  --track -n $4
              ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i $rho 0 $opt
            done
          done
        done
      done
  elif [ "$1" = "a2c" ];then
    for env in ${env_a2c[*]};do
      for opt in ${Optimizer[*]};do
        for rho in ${Rho[*]};do
            for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
              do
              echo "Test rho: $rho"
              python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $3 -P --rho $rho  --track -n $4
              ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i $rho 0 $opt
            done
          done
        done
      done
  elif [ "$1" = "sac" ];then
    for env in ${env_sac[*]};do
      for opt in ${Optimizer[*]};do
        for rho in ${Rho[*]};do
            for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
              do
              echo "Test rho: $rho"
              python train.py --algo $1 --env $env $2 --device cuda --optimize-choice $opt --quantize $3 -P --rho $rho  --track -n $4
              ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i $rho 0 $opt
            done
          done
        done
      done
    fi


# 3.Grid search lr, keep using rho 0.05
elif [ "$5" = "search_lr" ];then
  if [ "$1" = "dqn" ];then
    for env in ${env_dqn[*]};do
      for opt in ${Optimizer[*]};do
        for lr in ${Learning_rate[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env $2 --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate:$lr --track -n $4
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i 0.05 $lr $opt
          done
        done
      done
    done
  elif [ "$1" = "a2c" ];then
    for env in ${env_a2c[*]};do
      for opt in ${Optimizer[*]};do
        for lr in ${Learning_rate[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate:$lr --track -n $4
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i 0.05 $lr $opt
          done
        done
      done
    done
  elif [ "$1" = "sac" ];then
    for env in ${env_sac[*]};do
      for opt in ${Optimizer[*]};do
        for lr in ${Learning_rate[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate:$lr --track -n $4
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$i" $opt $i 0.05 $lr $opt
          done
        done
      done
    done
  fi
fi




