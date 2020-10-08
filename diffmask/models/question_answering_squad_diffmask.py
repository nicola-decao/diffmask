import torch
import numpy as np
import pytorch_lightning as pl
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from .question_answering_squad import (
    QuestionAnsweringSquad,
    BertQuestionAnsweringSquad,
)
from .gates import (
    DiffMaskGateInput,
    DiffMaskGateHidden,
    PerSampleDiffMaskGate,
    PerSampleREINFORCEGate,
    MLPMaxGate,
    MLPGate,
)
from ..optim.lookahead import LookaheadRMSprop
from ..utils.getter_setter import (
    bert_getter,
    bert_setter,
)
from ..utils.util import accuracy_precision_recall_f1


class QuestionAnsweringSquadDiffMask(QuestionAnsweringSquad):
    def __init__(self, hparams):
        super().__init__(hparams)

        for p in self.parameters():
            p.requires_grad_(False)

    def training_step(self, batch, batch_idx=None, optimizer_idx=None):

        if self.training and self.hparams.stop_train and self.hparams.layer_pred != -1:
            if (
                self.running_acc[self.hparams.layer_pred] > 0.75
                and self.running_l0[self.hparams.layer_pred] < 0.05
                and self.running_steps[self.hparams.layer_pred] > 1000
            ):
                return {"loss": torch.tensor(0.0, requires_grad=True)}

        (input_ids, mask, token_type_ids, start_positions, end_positions,) = batch

        (
            logits_start,
            logits_end,
            logits_start_orig,
            logits_end_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self.forward_explainer(input_ids, mask, token_type_ids)

        logits_start, logits_end, logits_start_orig, logits_end_orig = tuple(
            torch.where(mask.bool(), e, torch.full_like(e, -float("inf")))
            for e in (logits_start, logits_end, logits_start_orig, logits_end_orig)
        )

        loss_c = (
            torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=logits_start_orig),
                torch.distributions.Categorical(logits=logits_start),
            )
            + torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=logits_end_orig),
                torch.distributions.Categorical(logits=logits_end),
            )
            - self.hparams.eps
        )

        loss_g = (expected_L0 * mask).sum(-1) / mask.sum(-1)

        loss = self.alpha[layer_pred] * loss_c + loss_g

        acc, _, _, _ = accuracy_precision_recall_f1(
            torch.cat((logits_start.argmax(-1), logits_end.argmax(-1)), -1),
            torch.cat((logits_start_orig.argmax(-1), logits_end_orig.argmax(-1)), -1),
        )

        l0 = (expected_L0.exp() * mask).sum(-1) / mask.sum(-1)

        outputs_dict = {
            "loss_c": loss_c.mean(-1),
            "loss_g": loss_g.mean(-1),
            "alpha": self.alpha[layer_pred].mean(-1),
            "acc": acc,
            "l0": l0.mean(-1),
            "layer_pred": layer_pred,
            "r_acc": self.running_acc[layer_pred],
            "r_l0": self.running_l0[layer_pred],
            "r_steps": self.running_steps[layer_pred],
        }

        outputs_dict = {
            "loss": loss.mean(-1),
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        if self.training:
            self.running_acc[layer_pred] = (
                self.running_acc[layer_pred] * 0.9 + acc * 0.1
            )
            self.running_l0[layer_pred] = (
                self.running_l0[layer_pred] * 0.9 + l0.mean(-1) * 0.1
            )
            self.running_steps[layer_pred] += 1

        return outputs_dict

    def validation_epoch_end(self, outputs):

        outputs_dict = {
            k: [e[k] for e in outputs if k in e]
            for k in ("val_loss_c", "val_loss_g", "val_acc", "val_l0")
        }

        outputs_dict = {k: sum(v) / len(v) for k, v in outputs_dict.items()}

        outputs_dict["val_loss_c"] += self.hparams.eps

        outputs_dict = {
            "val_loss": outputs_dict["val_l0"]
            if outputs_dict["val_loss_c"] <= self.hparams.eps_valid
            and outputs_dict["val_acc"] >= self.hparams.acc_valid
            else torch.full_like(outputs_dict["val_l0"], float("inf")),
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        optimizers = [
            LookaheadRMSprop(
                params=[
                    {
                        "params": self.gate.g_hat.parameters(),
                        "lr": self.hparams.learning_rate,
                    },
                    {
                        "params": self.gate.placeholder.parameters()
                        if isinstance(self.gate.placeholder, torch.nn.ParameterList)
                        else [self.gate.placeholder],
                        "lr": self.hparams.learning_rate_placeholder,
                    },
                ],
                centered=True,
            ),
            LookaheadRMSprop(
                params=[self.alpha]
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha.parameters(),
                lr=self.hparams.learning_rate_alpha,
            ),
        ]

        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 24 * 50),
                "interval": "step",
            },
            get_constant_schedule(optimizers[1]),
        ]
        return optimizers, schedulers

    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            for i in range(len(self.alpha)):
                if self.alpha[i].grad:
                    self.alpha[i].grad *= -1

            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            for i in range(len(self.alpha)):
                self.alpha[i].data = torch.where(
                    self.alpha[i].data < 0,
                    torch.full_like(self.alpha[i].data, 0),
                    self.alpha[i].data,
                )
                self.alpha[i].data = torch.where(
                    self.alpha[i].data > 200,
                    torch.full_like(self.alpha[i].data, 200),
                    self.alpha[i].data,
                )


class BertQuestionAnsweringSquadDiffMask(
    QuestionAnsweringSquadDiffMask, BertQuestionAnsweringSquad,
):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(()))
                for _ in range(self.net.config.num_hidden_layers + 2)
            ]
        )

        gate = DiffMaskGateInput if self.hparams.gate == "input" else DiffMaskGateHidden

        self.gate = gate(
            hidden_size=self.net.config.hidden_size,
            hidden_attention=self.net.config.hidden_size // 4,
            num_hidden_layers=self.net.config.num_hidden_layers + 2,
            max_position_embeddings=1,
            gate_fn=MLPMaxGate if self.hparams.gate == "input" else MLPGate,
            gate_bias=hparams.gate_bias,
            placeholder=hparams.placeholder,
            init_vector=self.net.bert.embeddings.word_embeddings.weight[
                self.tokenizer.mask_token_id
            ]
            if self.hparams.layer_pred == 0 or self.hparams.gate == "input"
            else None,
        )

        self.register_buffer(
            "running_acc", torch.ones((self.net.config.num_hidden_layers + 2,))
        )
        self.register_buffer(
            "running_l0", torch.ones((self.net.config.num_hidden_layers + 2,))
        )
        self.register_buffer(
            "running_steps", torch.zeros((self.net.config.num_hidden_layers + 2,))
        )

    def forward_explainer(
        self,
        input_ids,
        mask,
        token_type_ids,
        start_positions=None,
        end_positions=None,
        layer_pred=None,
        attribution=False,
    ):

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": mask,
            "token_type_ids": token_type_ids,
        }

        self.net.eval()

        (logits_start_orig, logits_end_orig,), hidden_states = bert_getter(
            self.net, inputs_dict
        )

        if layer_pred is None:
            if self.hparams.layer_pred == -1:
                if self.hparams.stop_train:
                    criterion = lambda i: (
                        self.running_acc[i] > 0.75
                        and self.running_l0[i] < 0.1
                        and self.running_steps[i] > 100
                    )
                    p = np.array(
                        [0.1 if criterion(i) else 1 for i in range(len(hidden_states))]
                    )
                    layer_pred = np.random.choice(
                        range(len(hidden_states)), (), p=p / p.sum()
                    ).item()
                else:
                    layer_pred = torch.randint(len(hidden_states), ()).item()
            else:
                layer_pred = self.hparams.layer_pred

        if "hidden" in self.hparams.gate:
            layer_drop = layer_pred
        else:
            layer_drop = 0

        (
            new_hidden_state,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        ) = self.gate(
            hidden_states=hidden_states,
            mask=mask,
            layer_pred=None if attribution else layer_pred,
        )

        if attribution:
            return logits_start_orig, logits_end_orig, expected_L0_full
        else:

            new_hidden_states = (
                [None] * layer_drop
                + [new_hidden_state]
                + [None] * (len(hidden_states) - layer_drop - 1)
            )

            (logits_start, logits_end,), _ = bert_setter(
                self.net, inputs_dict, hidden_states=new_hidden_states,
            )

        return (
            logits_start,
            logits_end,
            logits_start_orig,
            logits_end_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        )


class PerSampleBertQuestionAnsweringSquadDiffMask(BertQuestionAnsweringSquadDiffMask):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.gate = PerSampleDiffMaskGate(
            hidden_size=self.net.config.hidden_size,
            num_hidden_layers=1,
            max_position_embeddings=384,
            batch_size=1,
            placeholder=False,
            init_vector=None,
        )

        self.alpha = torch.nn.Parameter(torch.ones((1,)))

    def prepare_data(self):
        # assign to use in dataloaders
        samples = [int(e) for e in self.hparams.samples.split(",")]
        if (
            not hasattr(self, "train_dataset")
            or not hasattr(self, "train_dataset_orig")
        ) and self.training:
            self.train_dataset, self.train_dataset_orig = self._squad_reader(
                self.hparams.train_filename,
            )

            self.train_dataset_orig = list(self.train_dataset_orig)
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, samples)
            self.train_dataset_orig = {
                self.train_dataset_orig[i][0]: self.train_dataset_orig[i][1]
                for i in samples
            }

        if not hasattr(self, "val_dataset") or not hasattr(self, "val_dataset_orig"):
            self.val_dataset, self.val_dataset_orig = self._squad_reader(
                self.hparams.val_filename,
            )

            self.val_dataset_orig = list(self.val_dataset_orig)
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, samples)
            self.val_dataset_orig = {
                self.val_dataset_orig[i][0]: self.val_dataset_orig[i][1]
                for i in samples
            }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=False
        )

    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

            self.gate.logits.data = torch.where(
                self.gate.logits.data < -10,
                torch.full_like(self.gate.logits.data, -10),
                self.gate.logits.data,
            )
            self.gate.logits.data = torch.where(
                self.gate.logits.data > 10,
                torch.full_like(self.gate.logits.data, 10),
                self.gate.logits.data,
            )
            self.gate.logits.data = torch.where(
                torch.isnan(self.gate.logits.data),
                torch.full_like(self.gate.logits.data, 0),
                self.gate.logits.data,
            )

        elif optimizer_idx == 1:
            self.alpha.grad *= -1
            optimizer.step()
            optimizer.zero_grad()

            self.alpha.data = torch.where(
                self.alpha.data < 0,
                torch.full_like(self.alpha.data, 0),
                self.alpha.data,
            )
            self.alpha.data = torch.where(
                self.alpha.data > 200,
                torch.full_like(self.alpha.data, 200),
                self.alpha.data,
            )
            self.alpha.data = torch.where(
                torch.isnan(self.alpha.data),
                torch.full_like(self.alpha.data, 1),
                self.alpha.data,
            )
