#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import torch
import numpy as np
import pytorch_lightning as pl

from diffmask.models.question_answering_squad_diffmask import (
    BertQuestionAnsweringSquadDiffMask,
    PerSampleBertQuestionAnsweringSquadDiffMask,
)
from diffmask.utils.callbacks import CallbackSquadDiffMask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--model",
        type=str,
        default="bert-large-uncased-whole-word-masking-finetuned-squad",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--train_filename",
        type=str,
        default="./datasets/squad/train-v1.1_bert-large-uncased-whole-word-masking-finetuned-squad.json",
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        default="./datasets/squad/dev-v1.1_bert-large-uncased-whole-word-masking-finetuned-squad.json",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--gate_bias", action="store_true")
    parser.add_argument("--learning_rate_alpha", type=float, default=3e-1)
    parser.add_argument("--learning_rate_placeholder", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--eps_valid", type=float, default=3)
    parser.add_argument("--acc_valid", type=float, default=0.0)
    parser.add_argument("--placeholder", action="store_true")
    parser.add_argument("--stop_train", action="store_true")
    parser.add_argument(
        "--gate",
        type=str,
        default="input",
        choices=["input", "hidden", "per_sample-reinforce", "per_sample-diffmask"],
    )
    parser.add_argument("--layer_pred", type=int, default=-1)
    
    hparams= parser.parse_args()
    
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu

    model = BertQuestionAnsweringSquadDiffMask(hparams)

    trainer = pl.Trainer(
        gpus=int(hparams.gpu != ""),
        progress_bar_refresh_rate=1,
        max_epochs=hparams.epochs,
        callbacks=[CallbackSquadDiffMask()],
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                "outputs",
                "squad-bert-{}-layer_pred={}".format(hparams.gate, hparams.layer_pred),
                "{epoch}-{val_acc:.2f}-{val_f1:.2f}-{val_l0:.2f}",
            ),
            verbose=True,
            save_top_k=50,
        ),
    )

    trainer.fit(model)
