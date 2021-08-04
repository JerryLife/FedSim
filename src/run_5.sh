#!/usr/bin/env bash

python src/train_song_fedsim.py -k 40 -g 2 > out/song/knn/song_fedsim_k_40_0.out
python src/train_song_fedsim.py -k 40 -g 2 > out/song/knn/song_fedsim_k_40_1.out
python src/train_song_fedsim.py -k 40 -g 2 > out/song/knn/song_fedsim_k_40_2.out
python src/train_song_fedsim.py -k 40 -g 2 > out/song/knn/song_fedsim_k_40_3.out
python src/train_song_fedsim.py -k 40 -g 2 > out/song/knn/song_fedsim_k_40_4.out
