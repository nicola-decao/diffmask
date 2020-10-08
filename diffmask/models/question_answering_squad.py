import json
from collections import OrderedDict
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from ..utils.util import accuracy_precision_recall_f1


def load_squad(path, tokenizer):

    with open(path, "r") as f:
        data = json.load(f)

    input_dicts = OrderedDict()
    data_orig = OrderedDict()
    for k, v in tqdm(data.items()):
        input_dict = tokenizer.encode_plus(
            v["question"],
            v["context"],
            max_length=384,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        input_dict["start_positions"] = torch.tensor(
            [e[0] + len(v["question"]) + 2 for e in v["answer_offsets"]]
        )
        input_dict["end_positions"] = torch.tensor(
            [e[1] + len(v["question"]) + 2 for e in v["answer_offsets"]]
        )

        if len(v["question"]) + len(v["context"]) + 2 < 384 and all(
            f < 384 for e in v["answer_offsets"] for f in e
        ):
            input_dicts[k] = input_dict
            data_orig[k] = v

    max_ans = max(t["end_positions"].shape[0] for t in input_dicts.values())

    tensor_dataset = [
        torch.cat([t[k] for t in input_dicts.values()], 0)
        for k in ("input_ids", "attention_mask", "token_type_ids",)
    ] + [
        torch.stack(
            [
                torch.nn.functional.pad(t[k], (0, max_ans - t[k].shape[0]), value=-1)
                for t in input_dicts.values()
            ],
            0,
        )
        for k in ("start_positions", "end_positions",)
    ]

    return torch.utils.data.TensorDataset(*tensor_dataset), data_orig


class QuestionAnsweringSquad(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model)

    def prepare_data(self):
        # assign to use in dataloaders
        if (
            not hasattr(self, "train_dataset")
            or not hasattr(self, "train_dataset_orig")
        ) and self.training:
            self.train_dataset, self.train_dataset_orig = load_squad(
                self.hparams.train_filename, self.tokenizer,
            )
        if not hasattr(self, "val_dataset") or not hasattr(self, "val_dataset_orig"):
            self.val_dataset, self.val_dataset_orig = load_squad(
                self.hparams.val_filename, self.tokenizer,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size
        )

    def training_step(self, batch, batch_idx=None):
        if self.training:
            return {
                "loss": torch.tensor(0.0, device=batch[0].device, requires_grad=True)
            }
        input_ids, mask, token_type_ids, start_positions, end_positions = batch

        logits_start, logits_end = self.forward(input_ids, mask, token_type_ids)[:2]

        loss_ = []
        acc_ = []
        for start_positions, end_positions in zip(start_positions.T, end_positions.T):

            loss = 0.5 * (
                torch.nn.functional.cross_entropy(
                    logits_start,
                    start_positions,
                    reduction="mean",
                    size_average=True,
                    ignore_index=-1,
                )
                + torch.nn.functional.cross_entropy(
                    logits_end,
                    end_positions,
                    reduction="mean",
                    size_average=True,
                    ignore_index=-1,
                )
            )

            acc, _, _, _ = accuracy_precision_recall_f1(
                torch.cat((logits_start.argmax(-1), logits_end.argmax(-1)), -1),
                torch.cat((start_positions, end_positions), -1),
            )

            loss_.append(loss)
            acc_.append(acc)

        loss = sum(loss_) / len(loss_)
        acc = max(acc_)

        outputs_dict = {
            "acc": acc,
        }

        outputs_dict = {
            "loss": loss,
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        return outputs_dict

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        acc = sum(e["val_acc"] for e in outputs) / len(outputs)

        outputs_dict = {
            "val_acc": acc,
        }

        outputs_dict = {
            "val_loss": -acc,
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.parameters(), self.hparams.learning_rate),
        ]
        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 200),
                "interval": "step",
            },
        ]

        return optimizers, schedulers


class BertQuestionAnsweringSquad(QuestionAnsweringSquad):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.net = BertForQuestionAnswering.from_pretrained(self.hparams.model)

    def forward(
        self, input_ids, mask, token_type_ids, start_positions=None, end_positions=None
    ):
        return self.net(
            input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids
        )
