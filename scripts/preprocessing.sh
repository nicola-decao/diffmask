#!/bin/bash

python scripts/preprocessing_sst.py \
    --model bert-base-uncased \
    --file_input ./datasets/sst/dev.txt \
    --file_output ./datasets/sst/dev-mapping.json

python scripts/preprocessing_squad.py \
    --model bert-large-uncased-whole-word-masking-finetuned-squad \
    --file_input ./datasets/squad/dev-v1.1.json \
    --file_output ./datasets/squad/dev-v1.1_bert-large-uncased-whole-word-masking-finetuned-squad.json

python scripts/preprocessing_squad.py \
    --model bert-large-uncased-whole-word-masking-finetuned-squad \
    --file_input ./datasets/squad/train-v1.1.json \
    --file_output ./datasets/squad/train-v1.1_bert-large-uncased-whole-word-masking-finetuned-squad.json
