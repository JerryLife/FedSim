#!/usr/bin/env bash




gpu=2

for dataset in ny hdb song game; do
  for i in  $(seq 0 $(($1 - 1))); do
    for algo in fedsim avgsim featuresim top1sim; do
      for k in 5 10 20; do
        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
          python src/train_"$dataset"_"$algo".py -s 0.2 -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
        else python src/train_"$dataset"_"$algo".py -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
        fi
      done
    done
  done
done

for dataset in beijing syn frog boone; do
  for i in  $(seq 0 $(($1 - 1))); do
    for algo in fedsim avgsim featuresim top1sim; do
      for k in 10 20 40; do
        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
          python src/train_"$dataset"_"$algo".py -s 0.2 -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
        else python src/train_"$dataset"_"$algo".py -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
        fi
      done
    done
  done
done


