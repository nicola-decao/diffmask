import re
import torch
import pytorch_lightning as pl
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from ..utils.util import accuracy_precision_recall_f1


def load_sst(path, tokenizer, lower=False):
    dataset_orig = []

    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\\", "")
            line = line.lower() if lower else line
            line = re.sub("\\\\", "", line)  # fix escape
            tokens = re.findall(r"\([0-9] ([^\(\)]+)\)", line)
            label = int(line[1])
            token_labels = list(map(int, re.findall(r"\(([0-9]) [^\(\)]", line)))
            assert len(tokens) == len(token_labels), "mismatch tokens/labels"
            dataset_orig.append((" ".join(tokens), label, token_labels))

    dataset = [(tokenizer.encode(s), t) for s, t, tl in dataset_orig]
    dataset = [(s + [0] * (84 - len(s)), t) for s, t in dataset]
    tensor_dataset = (
        torch.tensor([s for s, t in dataset]),
        (torch.tensor([s for s, t in dataset]) != 0).long(),
        torch.tensor([t for s, t in dataset]),
    )

    return torch.utils.data.TensorDataset(*tensor_dataset), dataset_orig


class SentimentClassificationSST(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model)

    def prepare_data(self):
        # assign to use in dataloaders
        if not hasattr(self, "train_dataset") or not hasattr(
            self, "train_dataset_orig"
        ):
            self.train_dataset, self.train_dataset_orig = load_sst(
                self.hparams.train_filename, self.tokenizer,
            )
        if not hasattr(self, "val_dataset") or not hasattr(self, "val_dataset_orig"):
            self.val_dataset, self.val_dataset_orig = load_sst(
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
        input_ids, mask, labels = batch

        logits = self.forward(input_ids, mask)[0]

        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(
            -1
        )

        acc, _, _, f1 = accuracy_precision_recall_f1(
            logits.argmax(-1), labels, average=True
        )

        outputs_dict = {
            "acc": acc,
            "f1": f1,
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

        outputs_dict = {
            k: sum(e[k] for e in outputs) / len(outputs) for k in ("val_acc", "val_f1")
        }

        outputs_dict = {
            "val_loss": -outputs_dict["val_f1"],
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


class BertSentimentClassificationSST(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        config = BertConfig.from_pretrained(self.hparams.model)
        config.num_labels = 5
        self.net = BertForSequenceClassification.from_pretrained(
            self.hparams.model, config=config
        )

    def forward(self, input_ids, mask, labels=None):
        return self.net(input_ids=input_ids, attention_mask=mask)


class RecurrentSentimentClassificationSST(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.emb = BertForSequenceClassification.from_pretrained(
            hparams.model
        ).bert.embeddings.word_embeddings.requires_grad_(False)

        self.gru = torch.nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=self.emb.embedding_dim,
            batch_first=True,
        )

        self.classifier = torch.nn.Linear(self.emb.embedding_dim, 5)

    def forward(self, input_ids, mask, labels=None):
        x = self.emb(input_ids)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(-1), batch_first=True, enforce_sorted=False
        )

        _, h = self.gru(x)

        return (self.classifier(h[0]),)
