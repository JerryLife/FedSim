#!/usr/bin/env bash

python src/train_song_avgsim.py -k 30 -g 0 > out/song/knn/song_avgsim_k_30_0.out
python src/train_song_avgsim.py -k 30 -g 0 > out/song/knn/song_avgsim_k_30_1.out
python src/train_song_avgsim.py -k 30 -g 0 > out/song/knn/song_avgsim_k_30_2.out
python src/train_song_avgsim.py -k 30 -g 0 > out/song/knn/song_avgsim_k_30_3.out
python src/train_song_avgsim.py -k 30 -g 0 > out/song/knn/song_avgsim_k_30_4.out
