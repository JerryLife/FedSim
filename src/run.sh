#!/usr/bin/env bash

for i in  $(seq 0 4); do
      python src/train_game_A.py > out/game/game_A_"$i".out &
      sleep 1
  done
  wait

for name in "game_fedsim" "game_avgsim" "game_top1sim"; do
  for i in  $(seq 0 2); do
      python src/train_"$name".py > out/game/"$name"_"$i".out &
      sleep 1
  done
  wait
done

for name in "game_fedsim" "game_avgsim" "game_top1sim"; do
  for i in  $(seq 3 4); do
      python src/train_"$name".py > out/game/"$name"_"$i".out &
      sleep 1
  done
  wait
done