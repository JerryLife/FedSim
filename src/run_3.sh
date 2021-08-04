#!/usr/bin/env bash

python src/train_song_fedsim.py -k 20 -g 1 > out/song/knn/song_fedsim_k_20_0.out
python src/train_song_fedsim.py -k 20 -g 1 > out/song/knn/song_fedsim_k_20_1.out
python src/train_song_fedsim.py -k 20 -g 1 > out/song/knn/song_fedsim_k_20_2.out
python src/train_song_fedsim.py -k 20 -g 1 > out/song/knn/song_fedsim_k_20_3.out
python src/train_song_fedsim.py -k 20 -g 1 > out/song/knn/song_fedsim_k_20_4.out
