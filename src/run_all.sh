#!/usr/bin/env bash

# ./run_all <number of rounds>
# E.g., ./run_all 10
for i in  $(seq 0 $(($1 - 1))); do
  for noise_scale in 0.00 0.10 0.20; do
    for name in "boone_all" "boone_A" "boone_avgsim" "boone_mergesim" "boone_concatsim" \
    "boone_top1sim" "boone_ordersim"; do
      python src/train_"$name".py -s="$noise_scale" > out/boone/"$name"_noise_"$noise_scale"_"$i".out &
      sleep 1
    done
    wait
    echo "MiniBooNE round ""$i"" noise $noise_scale done"

    for name in "syn_all" "syn_A" "syn_avgsim" "syn_mergesim" "syn_concatsim" \
    "syn_top1sim" "syn_ordersim"; do
      python src/train_"$name".py -s="$noise_scale" > out/syn/"$name"_noise_"$noise_scale"_"$i".out &
      sleep 1
    done
    wait
    echo "Sklearn-Synthetic round ""$i"" noise $noise_scale done"
  done

  for name in "beijing_avgsim" "beijing_house" "beijing_concatsim" "beijing_mergesim" \
   "beijing_ordersim" "beijing_top1sim"; do
      python src/train_"$name".py > out/beijing/"$name"_noise_"$noise_scale"_"$i".out &
      sleep 1
  done
  wait
  echo "Beijing House-Airbnb round ""$i"" done"

  # run ny sequentially due to high cost of memory
  for name in "ny_bike" "ny_top1sim" "ny_avgsim" "ny_ordersim" \
   "ny_concatsim" "ny_mergesim"; do
      python src/train_"$name".py > out/ny/"$name"_noise_"$noise_scale"_"$i".out
      sleep 1
  done
  echo "NY Bike-Taxi round ""$i"" done"
done