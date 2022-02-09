#!/usr/bin/env bash



gpu=3
for dataset in song; do
  mkdir -p out/performance/"$dataset"/no_priv/
  for i in  $(seq 0 $(($1 - 1))); do
    for algo in disable_sort; do
      python src/train_"$dataset"_fedsim.py -ds -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out
    done
    for algo in disable_weight; do
      python src/train_"$dataset"_fedsim.py -dw -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out
    done
  done
  wait
done