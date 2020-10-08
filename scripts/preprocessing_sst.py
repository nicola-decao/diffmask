#!/usr/bin/env python
# coding: utf-8


import argparse
import json
import spacy
from tqdm.auto import tqdm, trange
from transformers import BertTokenizerFast
from diffmask.models.sentiment_classification_sst import load_sst


def get_offsets_spaces(sent):
    offsets_spaces = []
    i, j = 0, 0
    while j < len(sent):
        if sent[j] == " ":
            offsets_spaces.append((i, j))
            i = j + 1
        j += 1
    offsets_spaces.append((i, j))
    return offsets_spaces


def pre_processing(tokenizer, file_input, file_output):

    _, dataset_orig = load_sst(file_input, tokenizer)

    for i in trange(len(dataset_orig)):
        sent, _, token_labels = dataset_orig[i]
        offsets_mapping = tokenizer.encode_plus(sent, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        token_labels_offset = [None for _ in offsets_mapping]
        offsets_spaces = get_offsets_spaces(sent)

        for (start_char, end_char), l in zip(offsets_spaces, token_labels):
            i_start = [j for j, e in enumerate(offsets_mapping) if e[0] == start_char][
                0
            ]
            i_end = [j for j, e in enumerate(offsets_mapping) if e[1] == end_char][0]

            for j in range(i_start, i_end + 1):
                token_labels_offset[j] = l

        dataset_orig[i] = dataset_orig[i] + (token_labels_offset,)

        sent = nlp(sent)
        err = False
        token_offset = [None for _ in offsets_mapping]
        for w in sent:
            try:
                if w.text.strip() != "":
                    start_char = w.idx + (len(w.text) - len(w.text.lstrip()))
                    end_char = w.idx + len(w) - (len(w.text) - len(w.text.rstrip()))
                    i_start = [
                        i for i, e in enumerate(offsets_mapping) if e[0] == start_char
                    ][0]
                    i_end = [
                        i for i, e in enumerate(offsets_mapping) if e[1] == end_char
                    ][0]

                    for j in range(i_start, i_end + 1):
                        token_offset[j] = (
                            w.pos_,
                            w.tag_,
                            w.is_stop,
                            w.is_punct,
                        )
            except IndexError:
                err = True
        if not err:
            dataset_orig[i] = dataset_orig[i] + (token_offset,)

    with open(file_output, "w") as f:
        json.dump(dataset_orig, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--file_input", type=str, default="./datasets/sst/dev.txt")
    parser.add_argument(
        "--file_output", type=str, default="./datasets/sst/dev-mapping.json"
    )
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    tokenizer = BertTokenizerFast.from_pretrained(args.model, add_special_tokens=False)
    pre_processing(tokenizer, args.file_input, args.file_output)
