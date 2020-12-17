#!/usr/bin/env bash

# ./run <name> <noise scale> <rounds>
# E.g: ./runs syn_mergesim 0.1 10
for i in $(seq 1 $3); do
  nohup python src/train_"$1".py > out/"$1"_scale_"$2"_"$i".out &
  sleep 1
done