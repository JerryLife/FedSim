#!/usr/bin/env bash

python src/train_song_avgsim.py -k 50 -g 0 > out/song/knn/song_avgsim_k_50_0.out
python src/train_song_avgsim.py -k 50 -g 0 > out/song/knn/song_avgsim_k_50_1.out
python src/train_song_avgsim.py -k 50 -g 0 > out/song/knn/song_avgsim_k_50_2.out
python src/train_song_avgsim.py -k 50 -g 0 > out/song/knn/song_avgsim_k_50_3.out
python src/train_song_avgsim.py -k 50 -g 0 > out/song/knn/song_avgsim_k_50_4.out
