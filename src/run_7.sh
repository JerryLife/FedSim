#!/usr/bin/env bash

python src/train_song_featuresim.py -k 20 -g 0 > out/song/knn/song_featuresim_k_20_0.out
python src/train_song_featuresim.py -k 20 -g 0 > out/song/knn/song_featuresim_k_20_2.out
python src/train_song_featuresim.py -k 20 -g 0 > out/song/knn/song_featuresim_k_20_3.out
python src/train_song_featuresim.py -k 20 -g 0 > out/song/knn/song_featuresim_k_20_4.out
python src/train_song_featuresim.py -k 40 -g 0 > out/song/knn/song_featuresim_k_40_4.out