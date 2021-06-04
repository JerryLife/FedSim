#!/usr/bin/env bash

python src/train_ny_avgsim.py -k 5 -g 0 > out/ny/knn/ny_avgsim_k_5_2.out
python src/train_song_fedsim.py -k 5 -g 0 > out/song/knn/song_fedsim_k_5_0.out
python src/train_song_fedsim.py -k 5 -g 0 > out/song/knn/song_fedsim_k_5_1.out
python src/train_song_fedsim.py -k 5 -g 0 > out/song/knn/song_fedsim_k_5_2.out
python src/train_song_fedsim.py -k 5 -g 0 > out/song/knn/song_fedsim_k_5_3.out
python src/train_song_fedsim.py -k 5 -g 0 > out/song/knn/song_fedsim_k_5_4.out
