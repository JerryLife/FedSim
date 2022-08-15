#!/usr/bin/env bash

gpu=2
#for dataset in syn frog boone; do
#  mkdir -p out/performance/"$dataset"/no_priv/
#  for i in  $(seq 0 $(($1 - 1))); do
#    for noise in 0.0 0.1 0.2; do
#      for algo in fedsim avgsim featuresim top1sim A; do
#        python src/train_"$dataset"_"$algo".py -g $gpu -s $noise > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_noise_"$noise"_"$i".out &
#      done
#      wait
#    done
#  done
#done
#
#
#for dataset in beijing hdb game; do
#  mkdir -p out/performance/"$dataset"/no_priv/
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in fedsim avgsim featuresim top1sim A; do
#      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out &
#    done
#    wait
#  done
#done
#
#
#for dataset in game; do
#  mkdir -p out/performance/"$dataset"/no_priv/
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in exact; do
#      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out &
#    done
#  done
#  wait
#done




#for dataset in boone; do
#  mkdir -p out/performance/"$dataset"/no_priv/
#  for i in  $(seq 0 $(($1 - 1))); do
#    for noise in 0.1; do
#      for algo in fedsim avgsim featuresim top1sim A; do
#        python src/train_"$dataset"_"$algo".py -g $gpu -s $noise > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_noise_"$noise"_"$i".out &
#      done
#      wait
#    done
#  done
#done


for dataset in company; do
  mkdir -p out/performance/"$dataset"/no_priv/
  for i in  $(seq 0 $(($1 - 1))); do
    for algo in fedsim avgsim; do
      python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out &
    done
  done
  wait
done
