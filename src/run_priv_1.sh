#!/usr/bin/env bash




gpu=0
export PYTHONPATH="$(realpath src)"
for dataset in syn boone frog beijing hdb ny; do
  for i in  $(seq 0 $(($1 - 1))); do
    for algo in fedsim featuresim top1sim; do
      for tau in 1e0 1e-1 1e-2 1e-3 1e-4 5e-5 ; do
        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
          python src/priv_scripts/train_"$dataset"_"$algo".py -s 0.2 -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out
        else python src/priv_scripts/train_"$dataset"_"$algo".py -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out
        fi
      done
    done
  done
done



