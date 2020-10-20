#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import torch
import numpy as np
import pytorch_lightning as pl

from diffmask.models.sentiment_classification_sst import (
    BertSentimentClassificationSST,
    RecurrentSentimentClassificationSST,
)
from diffmask.models.sentiment_classification_sst_diffmask import (
    BertSentimentClassificationSSTDiffMask,
    RecurrentSentimentClassificationSSTDiffMask,
    PerSampleDiffMaskRecurrentSentimentClassificationSSTDiffMask,
    PerSampleREINFORCERecurrentSentimentClassificationSSTDiffMask,
)
from diffmask.utils.callbacks import CallbackSSTDiffMask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--train_filename", type=str, default="./datasets/sst/train.txt"
    )
    parser.add_argument("--val_filename", type=str, default="./datasets/sst/dev.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--architecture", type=str, default="bert", choices=["gru", "bert"]
    )
    parser.add_argument("--gate_bias", action="store_true")
    parser.add_argument(
        "--model_path", type=str, default="./outputs/sst-bert/version_0/checkpoints",
    )

    parser.add_argument("--learning_rate_alpha", type=float, default=3e-1)
    parser.add_argument("--learning_rate_placeholder", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--eps_valid", type=float, default=0.8)
    parser.add_argument("--acc_valid", type=float, default=0.75)
    parser.add_argument("--placeholder", action="store_true")
    parser.add_argument("--stop_train", action="store_true")
    parser.add_argument(
        "--gate",
        type=str,
        default="input",
        choices=["input", "hidden", "per_sample-diffmask", "per_sample-reinforce"],
    )
    parser.add_argument("--layer_pred", type=int, default=-1)
    
    hparams = parser.parse_args()

    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu

    hparams.model_path_ = os.path.join(
        hparams.model_path, os.listdir(hparams.model_path)[0]
    )

    if hparams.architecture == "bert":
        model_orig = BertSentimentClassificationSST.load_from_checkpoint(
            hparams.model_path_
        )
        model = BertSentimentClassificationSSTDiffMask(hparams)
    elif hparams.architecture == "gru":
        model_orig = RecurrentSentimentClassificationSST.load_from_checkpoint(
            hparams.model_path_
        )

        if "per_sample" not in hparams.gate:
            model = RecurrentSentimentClassificationSSTDiffMask(hparams)
        elif "diffmask" in hparams.gate:
            assert (
                hparams.batch_size == 1101
                and hparams.train_filename == hparams.val_filename
            )
            model = PerSampleDiffMaskRecurrentSentimentClassificationSSTDiffMask(
                hparams
            )
        elif "reinforce" in hparams.gate:
            assert (
                hparams.batch_size == 1101
                and hparams.train_filename == hparams.val_filename
            )
            model = PerSampleREINFORCERecurrentSentimentClassificationSSTDiffMask(
                hparams
            )
        else:
            raise RuntimeError
    else:
        raise RuntimeError

    model.load_state_dict(model_orig.state_dict(), strict=False)

    trainer = pl.Trainer(
        gpus=int(hparams.gpu != ""),
        progress_bar_refresh_rate=1 if hparams.architecture == "bert" else 10,
        max_epochs=hparams.epochs,
        check_val_every_n_epoch=1 if "per_sample" not in hparams.gate else 150,
        callbacks=[CallbackSSTDiffMask()],
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                "outputs",
                "sst-{}-{}".format(hparams.architecture, hparams.gate,),
                "{epoch}-{val_acc:.2f}-{val_f1:.2f}-{val_l0:.2f}",
            ),
            verbose=True,
            save_top_k=50,
        ),
    )

    trainer.fit(model)
