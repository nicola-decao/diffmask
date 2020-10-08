import math
import numpy as np
import torch
import pytorch_lightning as pl


class ToyTaskModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.embedding_query = torch.nn.Embedding(
            num_embeddings=hparams.num_embeddings, embedding_dim=hparams.embedding_dim
        )

        self.embedding_input = torch.nn.Embedding(
            num_embeddings=hparams.num_embeddings, embedding_dim=hparams.embedding_dim
        )

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hparams.embedding_dim * 3, hparams.hidden_size,),
            torch.nn.Tanh(),
            torch.nn.Linear(hparams.hidden_size, 2,),
        )

        self.gru = torch.nn.GRU(input_size=2, hidden_size=hparams.hidden_size,)

        self.head = torch.nn.Linear(hparams.hidden_size, 1,)

    def prepare_data(self):
        # generate data
        if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):

            torch.manual_seed(self.hparams.seed)
            np.random.seed(self.hparams.seed)

            query_ids = torch.tensor(
                [
                    np.random.choice(
                        self.hparams.num_embeddings, (2,), replace=False
                    ).tolist()
                    for _ in range(self.hparams.data_size)
                ]
            )

            input_ids = torch.tensor(
                [
                    np.random.choice(
                        self.hparams.num_embeddings,
                        (self.hparams.data_length),
                        p=[
                            0.25 if i in q else 0.5 / (self.hparams.num_embeddings - 2)
                            for i in range(self.hparams.num_embeddings)
                        ],
                    ).tolist()
                    for q in query_ids
                ]
            )

            mask = (
                (
                    torch.arange(self.hparams.num_embeddings) + 1
                    <= torch.randint(
                        1, self.hparams.num_embeddings + 1, (self.hparams.data_size, 1)
                    )
                )
            ).long()

            input_ids = torch.where(
                mask.bool(), input_ids, torch.full_like(input_ids, -1),
            )

            labels = torch.tensor(
                [
                    ((i == q[0]).sum(-1) > (i == q[1]).sum(-1)).item()
                    for q, i in zip(query_ids, input_ids)
                ]
            ).long()

            input_ids = torch.where(
                mask.bool(), input_ids, torch.full_like(input_ids, 0),
            )

            # assign to use in dataloaders
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                torch.utils.data.TensorDataset(query_ids, input_ids, mask, labels),
                [
                    math.floor(self.hparams.data_size * 0.9),
                    self.hparams.data_size - math.floor(self.hparams.data_size * 0.9),
                ],
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate,)

    def forward(self, query_ids, input_ids, mask, labels=None):
        q = self.embedding_query(query_ids)  # b x 2 x d
        x = self.embedding_input(input_ids)  # b x T x d

        q = q.flatten(-2, -1).unsqueeze(-2).repeat(1, x.shape[-2], 1)  # b x T x 2 * d

        x = self.encoder(torch.cat((q, x), -1))

        _, x = self.gru(
            torch.nn.utils.rnn.pack_padded_sequence(
                x, mask.sum(-1), batch_first=True, enforce_sorted=False
            )
        )

        return self.head(x[0]).squeeze(-1)

    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        query_ids, input_ids, mask, labels = batch

        logits = self.forward(query_ids, input_ids, mask)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        ).mean(-1) - 0.35 * torch.distributions.Categorical(
            logits=logits
        ).entropy().mean(
            -1
        )

        acc = ((logits > 0).float() == labels).float().mean(-1) * 100

        outputs_dict = {"acc": acc}

        outputs_dict = {
            k if self.training else "val_" + k: v for k, v in outputs_dict.items()
        }

        return {"loss": loss, **outputs_dict, "progress_bar": outputs_dict}

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        acc = sum(e["val_acc"] for e in outputs) / len(outputs)
        results = {"val_acc": acc, "val_loss": -acc}
        return results
