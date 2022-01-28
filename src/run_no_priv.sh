#!/usr/bin/env bash

gpu="1"
#for dataset in beijing ny hdb song game syn frog boone; do
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in fedsim avgsim featuresim top1sim A; do
#      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out
#    done
#  done
#done

#for dataset in song game; do
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in exact; do
#      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out
#    done
#  done
#done

for dataset in game; do
  for i in  $(seq 0 4); do
    for algo in fedsim avgsim featuresim top1sim A; do
      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out
    done
  done
done

for dataset in game song; do
  for i in  $(seq 0 4); do
    for algo in exact; do
      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out
    done
  done
done

for dataset in song; do
  for i in  $(seq 3 4); do
    for algo in fedsim avgsim featuresim top1sim A; do
      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out
    done
  done
done


