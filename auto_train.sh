#!/bin/bash
set -e
#########################################
#Positional Parameters
# Grid search of rho,learning-rate and lambda_hero
# $1 is the name of algorithm
# $2 is the set of quantization bit(QAT), using 32 bit means no QAT
# $3 is the test times(seeds number)
# $4 is timestep for training
# $5 is the choice of training, search all lr and rho, search lr , search rho
########################################

env_dqn=('CartPole-v1' 'MountainCar-v0' 'LunarLander-v2' 'Acrobot-v1')
env_a2c=('CartPole-v1' 'LunarLander-v2' 'MountainCar-v0' 'Acrobot-v1' 'Pendulum-v1' 'LunarLanderContinuous-v2')
#env_a2c=('CartPole-v1' 'LunarLander-v2' 'Pendulum-v1')
env_sac=('MountainCarContinuous-v0' 'Pendulum-v1' 'LunarLanderContinuous-v2')
Optimizer=("SAM" "base")
lambda_hero=(0.01 0.05 0.1 0.5 1.0 5.0)
Learning_rate=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)
Rho=(0.01 0.02 0.05 0.1 0.2 0.5)
echo "Grid search of Learning rate:[${Learning_rate[*]}]"
echo "Grid search of rho:[${Rho[*]}]"

# 1.Grid search lr and rho simultaneously by default
if [ -z "$5" ]||[ "$5" = "search_all" ];then
  if [ "$1" = "dqn" ];then
  for opt in ${Optimizer[*]};do
    num=1
    for env in ${env_dqn[*]};do
      for lambda in ${lambda_hero[*]};do
        for lr in ${Learning_rate[*]};do
          for rho in ${Rho[*]};do
            for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
              do
              echo "Test learning_rate: $lr, rho: $rho"
              python train.py --algo $1 --env $env  --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho -params learning_rate:$lr --track -n $4 --lambda_hero $lambda
              ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num $rho $lr
              ((num++))
            done
          done
        done
      done
    done
  done
  elif [ "$1" = "a2c" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_a2c[*]};do
        for lambda in ${lambda_hero[*]};do
          for lr in ${Learning_rate[*]};do
            for rho in ${Rho[*]};do
              for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
                do
                echo "Test learning_rate: $lr, rho: $rho"
                python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho -params learning_rate:$lr --track -n $4 --lambda_hero $lambda
                ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num $rho $lr
                ((num++))
              done
            done
          done
        done
      done
    done
  elif [ "$1" = "sac" ]; then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_sac[*]};do
        for lambda in ${lambda_hero[*]};do
          for lr in ${Learning_rate[*]};do
            for rho in ${Rho[*]};do
              for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
                do
                echo "Test learning_rate: $lr, rho: $rho"
                python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho -params learning_rate:$lr --track -n $4 --lambda_hero $lambda
                ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num $rho $lr
                ((num++))
              done
            done
          done
        done
      done
    done
  fi


# 2.Grid search rho, keep using lr suggested by default(rl_baselines3_zoo) and lambda 1.0
elif [ "$5" = "search_rho" ];then
  if [ "$1" = "dqn" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_dqn[*]};do
        for rho in ${Rho[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test rho: $rho"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho $rho  --track -n $4 --lambda_hero 1.0
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num $rho 0
            ((num++))
          done
        done
      done
    done

  elif [ "$1" = "a2c" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_a2c[*]};do
        for rho in ${Rho[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test rho: $rho"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $3 -P --rho $rho  --track -n $4 --lambda_hero 1.0
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num $rho 0
            ((num++))
          done
        done
      done
    done
  elif [ "$1" = "sac" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_sac[*]};do
        for rho in ${Rho[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test rho: $rho"
            python train.py --algo $1 --env $env $2 --device cuda --optimize-choice $opt --quantize $3 -P --rho $rho  --track -n $4 --lambda_hero 1.0
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num $rho 0
            ((num++))
          done
        done
      done
    done
  fi


# 3.Grid search lr, keep using rho 0.05 and lambda 1.0
elif [ "$5" = "search_lr" ];then
  if [ "$1" = "dqn" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_dqn[*]};do
        for lr in ${Learning_rate[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env $2 --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate:$lr --track -n $4 --lambda_hero 1.0
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num 0.05 $lr
            ((num++))
          done
        done
      done
    done
  elif [ "$1" = "a2c" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_a2c[*]};do
        for lr in ${Learning_rate[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate:$lr --track -n $4 --lambda_hero 1.0
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num 0.05 $lr
            ((num++))
          done
        done
      done
    done
  elif [ "$1" = "sac" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_sac[*]};do
        for lr in ${Learning_rate[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate:$lr --track -n $4 --lambda_hero 1.0
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num 0.05 $lr
            ((num++))
          done
        done
      done
    done
  fi

# 4.Grid search lambda_hero, keep using rho 0.05 and lr 0.01
elif [ "$5" = "search_lambda" ];then
  if [ "$1" = "dqn" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_dqn[*]};do
        for lambda in ${lambda_hero[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env $2 --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate: 0.01 --track -n $4 --lambda_hero $lambda
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num 0.05 0.01
            ((num++))
          done
        done
      done
    done
  elif [ "$1" = "a2c" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_a2c[*]};do
        for lambda in ${lambda_hero[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test learning_rate: $lr"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate: 0.01 --track -n $4 --lambda_hero $lambda
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num 0.05 0.01
            ((num++))
          done
        done
      done
    done
  elif [ "$1" = "sac" ];then
    for opt in ${Optimizer[*]};do
      num=1
      for env in ${env_sac[*]};do
        for lambda in ${lambda_hero[*]};do
          for ((i=1;i<=$3;i++)) # Test different random seeds, $5 can be set random number, but for fairness, it should be set big enough
            do
            echo "Test lambda_hero: $lambda"
            python train.py --algo $1 --env $env --device cuda --optimize-choice $opt --quantize $2 -P --rho 0.05 -params learning_rate: 0.01 --track -n $4 --lambda_hero $lambda
            ptq_all.sh $1 $env "logs/$1/"$env"_$2bit_"$opt"_$num" $opt $num 0.05 0.01
            ((num++))
          done
        done
      done
    done
  fi
fi




