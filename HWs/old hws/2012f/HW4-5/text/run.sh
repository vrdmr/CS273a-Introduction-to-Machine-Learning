#!/bin/bash

#----------------------------------------------------
# file:    run.sh
# purpose: run topic model end-to-end
# usage:   run.sh
# inputs:  docs.txt
# outputs: topics.txt, topicsindocs.txt
# author:  David Newman, newman@uci.edu
#----------------------------------------------------

### make all codes
make clean
make

### create wordstream.txt
./Makewordstream.pl < docs.txt > wordstream.txt

### create vocab.txt
./Makevocab.pl < wordstream.txt > vocab.txt

### create docword.txt
./Makedocword.pl < wordstream.txt > docword.txt

### run topic model (T=10 topics, iter=200 iterations), create Nwt.txt, Ndt.txt, z.txt
./topicmodel 10 200 777

### print topics.txt
./printtopics > topics.txt
#matlab -nojvm < printtopics.m

### print topicsindocs.txt
./printtopicsindocs > topicsindocs.txt
#matlab -nojvm < printtopicsindocs.m

cat topics.txt

date
ls -lrt *.txt

