#!/usr/bin/env python
# coding: utf-8


import os
import argparse

import torch
import pytorch_lightning as pl

from diffmask.models.sentiment_classification_sst import (
    BertSentimentClassificationSST,
    RecurrentSentimentClassificationSST,
)
from diffmask.utils import CallbackSST


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default="0")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--train_filename", type=str, default="./datasets/sst/train.txt"
    )
    parser.add_argument("--val_filename", type=str, default="./datasets/sst/dev.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument(
        "--architecture", type=str, default="bert", choices=["gru", "bert"]
    )

    hparams = parser.parse_args()

    torch.manual_seed(hparams.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu)

    if hparams.architecture == "bert":
        model = BertSentimentClassificationSST(hparams)
    elif hparams.architecture == "gru":
        model = RecurrentSentimentClassificationSST(hparams)
    else:
        raise RuntimeError

    trainer = pl.Trainer(
        gpus=int(hparams.gpu != ""),
        progress_bar_refresh_rate=1 if hparams.architecture == "bert" else 10,
        max_epochs=hparams.epochs,
        logger=pl.loggers.TensorBoardLogger(
            "outputs", name="sst-{}".format(hparams.architecture)
        ),
        callbacks=[CallbackSST()],
    )

    trainer.fit(model)
