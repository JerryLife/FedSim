#!/usr/bin/env bash

# ./run_all 10
for noise_scale in 0.00 0.10 0.20 0.30 0.40
do
#  name="boone_all"
#  echo "Training "$name
#  for i in $(seq 0 $(($1 - 1))); do
#    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
#    sleep 1
#  done
#  wait
#  echo $name" done."
#
#  name="boone_A"
#  echo "Training "$name
#  for i in $(seq 0 $(($1 - 1))); do
#    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
#    sleep 1
#  done
#  wait
#  echo $name" done."
#
  name="boone_avgsim"
  echo "Training "$name
  for i in $(seq 0 4); do
    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
    sleep 1
  done
  wait
  echo "Training "$name
  for i in $(seq 5 $(($1 - 1))); do
    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
    sleep 1
  done
  wait
  echo $name" done."

#  name="boone_mergesim"
#  echo "Training "$name
#  for i in $(seq 0 4); do
#    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
#    sleep 1
#  done
#  wait
#  echo $name" done."
#  echo "Training "$name
#  for i in $(seq 5 $(($1 - 1))); do
#    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
#    sleep 1
#  done
#  wait
#  echo $name" done."

#  name="boone_top1sim"
#  echo "Training "$name
#  for i in $(seq 0 4); do
#    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
#    sleep 1
#  done
#  wait
#  for i in $(seq 5 $(($1 - 1))); do
#    python src/train_"$name".py > out/"$name"_scale_"$noise_scale"_"$i".out &
#    sleep 1
#  done
#  wait
#  echo $name" done."
done