#!/bin/bash

set -e

#for ((i=1;i<=10;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 160000
#done

#echo "timestep 70000">rl_zoo3/data_all.txt

#for ((i=1;i<=10;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 70000
#done
#
#echo "timestep 80000">rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 80000
#done

#echo "timestep 90000">rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 90000
#done
#
#echo "timestep 100000">rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 100000
#done
#echo $(date) >> rl_zoo3/data_all.txt
#echo " SAM timestep 100000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 100000
#done
#echo $(date) >> rl_zoo3/data_all.txt
#
#echo " SAM timestep 110000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 110000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo " SAM timestep 120000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 120000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo " SAM timestep 130000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 130000
#done
#
#echo $(date)>> rl_zoo3/data_all.txt
#echo " SAM timestep 140000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 140000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo " SAM timestep 150000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 150000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo " SAM timestep 160000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 160000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo " SAM timestep 170000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 170000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo " SAM timestep 180000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 180000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo " SAM timestep 190000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -n 190000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
######

#echo "timestep base 60000">> rl_zoo3/data_all.txt
#echo "timestep 60000">> rl_zoo3/data_all.txt
#echo $(date)>> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 60000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo "timestep 70000">> rl_zoo3/data_all.txt
#echo $(date)>> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 70000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo "timestep 80000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 80000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo "timestep 90000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 90000
#done
#echo $(date)>> rl_zoo3/data_all.txt
#
#echo "timestep 100000">> rl_zoo3/data_all.txt
#for ((i=1;i<=15;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -n 100000
#done
#echo $(date)>> rl_zoo3/data_all.txt

#echo "timestep default 50000">> rl_zoo3/data_all.txt
#for ((i=1;i<=8;i++))
#do
#    python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 --seed 1
#done
#echo $(date)>> rl_zoo3/data_all.txt

echo "200k base">> rl_zoo3/data_all.txt
echo $(date)>> rl_zoo3/data_all.txt
python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice base --quantized 32 -tb tensorboard_log -n 2000000
echo $(date)>> rl_zoo3/data_all.txt

echo "200k base">> rl_zoo3/data_all.txt
echo $(date)>> rl_zoo3/data_all.txt
python  train.py --algo dqn --env CartPole-v1 --device cuda --optimize-choice SAM --quantized 32 -tb tensorboard_log  -n 2000000
echo $(date)>> rl_zoo3/data_all.txt

