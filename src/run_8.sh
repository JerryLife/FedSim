#!/usr/bin/env bash


python src/train_song_featuresim.py -k 10 -g 0 > out/song/knn/song_featuresim_k_10_2.out
python src/train_song_featuresim.py -k 10 -g 0 > out/song/knn/song_featuresim_k_10_3.out
python src/train_song_featuresim.py -k 10 -g 0 > out/song/knn/song_featuresim_k_10_4.out
python src/train_song_featuresim.py -k 30 -g 0 > out/song/knn/song_featuresim_k_30_3.out
python src/train_song_featuresim.py -k 30 -g 0 > out/song/knn/song_featuresim_k_30_4.out

