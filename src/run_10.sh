#!/usr/bin/env bash

python src/train_song_avgsim.py -k 10 -g 5 > out/song/knn/song_avgsim_k_10_0.out
python src/train_song_avgsim.py -k 10 -g 5 > out/song/knn/song_avgsim_k_10_1.out
python src/train_song_avgsim.py -k 10 -g 5 > out/song/knn/song_avgsim_k_10_2.out
python src/train_song_avgsim.py -k 10 -g 5 > out/song/knn/song_avgsim_k_10_3.out
python src/train_song_avgsim.py -k 10 -g 5 > out/song/knn/song_avgsim_k_10_4.out
