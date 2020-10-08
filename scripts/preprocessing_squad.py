#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def pre_processing(tokenizer, file_input, file_output):
    with open(file_input, "r") as f:
        data = json.load(f)["data"]

    new_data = {}
    for p in tqdm([p for d in data for p in d["paragraphs"]]):
        for qas in p["qas"]:
            question = tokenizer.tokenize(qas["question"])

            answer_offsets_ = set()
            for a in qas["answers"]:
                answer_offsets = (a["answer_start"], a["answer_start"] + len(a["text"]))
                context_pre_answer = tokenizer.tokenize(
                    p["context"][: answer_offsets[0]]
                )
                context_answer = tokenizer.tokenize(
                    p["context"][answer_offsets[0] : answer_offsets[1]]
                )
                context_post_answer = tokenizer.tokenize(
                    p["context"][answer_offsets[1] :]
                )
                context = context_pre_answer + context_answer + context_post_answer
                answer_offsets_.add(
                    (
                        len(context_pre_answer),
                        len(context_pre_answer) + len(context_answer) - 1,
                    )
                )

            new_data[qas["id"]] = {
                "question": question,
                "context": context,
                "answer_offsets": list(answer_offsets_),
            }

    with open(file_output, "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="bert-large-uncased-whole-word-masking-finetuned-squad",
    )
    parser.add_argument(
        "--file_input", type=str, default="./datasets/squad/dev-v1.1.json"
    )
    parser.add_argument(
        "--file_output",
        type=str,
        default="./datasets/squad/dev-v1.1_bert-large-uncased-whole-word-masking-finetuned-squad.json",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pre_processing(tokenizer, args.file_input, args.file_output)
