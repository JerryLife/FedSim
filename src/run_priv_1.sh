#!/usr/bin/env bash




gpu=1
export PYTHONPATH="$(realpath src)"

#for dataset in syn; do
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in fedsim featuresim top1sim avgsim; do
#      for tau in 1e-1 1e-2 1e-3 1e-4 5e-5 ; do
#        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
#          python src/priv_scripts/train_"$dataset"_"$algo".py -s 0.2 -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
#        else python src/priv_scripts/train_"$dataset"_"$algo".py -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
#        fi
#      done
#      wait
#    done
#  done
#done
#
#for dataset in boone; do
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in avgsim; do
#      for tau in 1e-1 1e-2 1e-3 1e-4 5e-5 ; do
#        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
#          python src/priv_scripts/train_"$dataset"_"$algo".py -s 0.2 -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
#        else python src/priv_scripts/train_"$dataset"_"$algo".py -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
#        fi
#      done
#      wait
#    done
#  done
#done

for dataset in game; do
  for i in  $(seq 0 $(($1 - 1))); do
    for algo in fedsim featuresim top1sim avgsim; do
      for tau in 5e-1 5e-2 5e-3 2e-3 ; do
        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
          python src/priv_scripts/train_"$dataset"_"$algo".py -s 0.2 -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
        else python src/priv_scripts/train_"$dataset"_"$algo".py -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
        fi
      done
      wait
    done
  done
done

for dataset in song; do
  for i in  $(seq 0 $(($1 - 1))); do
    for algo in fedsim featuresim top1sim avgsim; do
      for tau in 5e-1 5e-2 5e-3 2e-3 ; do
        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
          python src/priv_scripts/train_"$dataset"_"$algo".py -s 0.2 -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
        else python src/priv_scripts/train_"$dataset"_"$algo".py -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out &
        fi
      done
      wait
    done
  done
done

#
#for dataset in song ny; do
#  for i in  $(seq 0 $(($1 - 1))); do
#    for algo in fedsim featuresim top1sim avgsim; do
#      for tau in 1e-1 1e-2 1e-3 1e-4 5e-5 ; do
#        if [[ ( "$dataset" = "syn" ) || ( "$dataset" = "frog" ) || ( "$dataset" = "boone" ) ]]; then
#          python src/priv_scripts/train_"$dataset"_"$algo".py -s 0.2 -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out
#        else python src/priv_scripts/train_"$dataset"_"$algo".py -p $tau -g $gpu > out/performance/"$dataset"/priv/"$dataset"_"$algo"_p_"$tau"_"$i".out
#        fi
#      done
#    done
#  done
#done
#
#
#
#
