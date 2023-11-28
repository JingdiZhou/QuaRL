#!/bin/bash
#
#set -e
#
#for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 5 6 4 2; do
#    python new_ptq.py --algo $1 --env $2 --quantized $i --exp-id 1
#done
#python collate_model.py --algo $1 --env $2 --device cuda  -f $3

#after training by user
set -e

for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 5 6 4 2; do
    python new_ptq.py --algo $1 --env $2 --quantized $i --exp-id 1  -f $3 --optimize-choice $4
done
python collate_model.py --algo $1 --env $2 --device cuda  -f quantized/ --no-render
