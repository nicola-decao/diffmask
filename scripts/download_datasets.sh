#!/bin/bash

mkdir datasets
cd datasets

mkdir squad
cd squad

echo "Downloading SQuAD data .."
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

cd ..
mkdir sst
cd sst

echo "Downloading SST data .."
wget https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip -j trainDevTestTrees_PTB.zip
rm trainDevTestTrees_PTB.zip

cd ../../
echo "Done!"
