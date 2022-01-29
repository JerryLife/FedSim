#!/usr/bin/env bash




gpu=7

for dataset in ny; do
  mkdir -p out/performance/"$dataset"/knn/
  for i in $(seq 0 $(($1 - 1))); do
    for k in 3 5 40; do
      for algo in fedsim avgsim featuresim top1sim; do
        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
          python src/train_"$dataset"_"$algo".py -s 0.2 -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
        else python src/train_"$dataset"_"$algo".py -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
        fi
      done
    done
  done
done


#for dataset in ny hdb song game; do
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in fedsim avgsim featuresim top1sim; do
#      for k in 5 10 20; do
#        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
#          python src/train_"$dataset"_"$algo".py -s 0.2 -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
#        else python src/train_"$dataset"_"$algo".py -k $k -g $gpu > out/performance/"$dataset"/knn/"$dataset"_"$algo"_k_"$k"_"$i".out
#        fi
#      done
#    done
#  done
#done



#for dataset in syn boone frog; do
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in A; do
#      mkdir -p out/performance/"$dataset"/no_priv
#      if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
#        python src/train_"$dataset"_"$algo".py -s 0.2 -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out &
#      else python src/train_"$dataset"_"$algo".py -g $gpu > out/performance/"$dataset"/no_priv/"$dataset"_"$algo"_"$i".out &
#      fi
#    done
#    wait
#  done
#done


