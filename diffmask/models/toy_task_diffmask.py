import torch
from diffmask.models.toy_task import ToyTaskModel
from diffmask.models.gates import MLPGate, MLPMaxGate
from diffmask.models.distributions import RectifiedStreched, BinaryConcrete
from diffmask.optim.lookahead import LookaheadRMSprop
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from ..utils.getter_setter import (
    toy_getter,
    toy_setter,
)


class ToyTaskModelDiffMask(ToyTaskModel):
    def __init__(self, hparams):
        super().__init__(hparams)

        for p in super().parameters():
            p.requires_grad_(False)

        self.gate = torch.nn.ModuleList(
            [
                MLPMaxGate(hparams.hidden_size + hparams.hidden_size, 32),
                MLPMaxGate(2 + 2, 32),
            ]
        )
        self.placeholder = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(1, 2, 1, hparams.hidden_size,))
        )

        self.alpha = torch.nn.Parameter(torch.ones((2,)))

    def forward_explainer(
        self, query_ids, input_ids, mask, layer_drop=None, layer_pred=None
    ):

        inputs_dict = {
            "query_ids": query_ids,
            "input_ids": input_ids,
            "mask": mask,
        }

        with torch.no_grad():
            logits_orig, hidden_states = toy_getter(self, inputs_dict)

        if layer_drop is None:
            layer_drop = torch.randint(2, ()).item()

        if layer_pred is None:
            layer_pred = layer_drop

        logits = self.gate[layer_drop](
            hidden_states[1 + layer_drop], hidden_states[1 + layer_pred],
        ).squeeze(-1)

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )

        gates_full = dist.rsample()
        expected_L0_full = dist.log_expected_L0()

        gates = gates_full
        expected_L0 = expected_L0_full

        inputs_dict = {
            "query_ids": query_ids,
            "input_ids": input_ids,
            "mask": mask,
        }

        logits, _ = toy_setter(
            self,
            inputs_dict,
            (
                [None] * (layer_drop + 1)
                + [
                    hidden_states[layer_drop + 1] * gates.unsqueeze(-1)
                    + self.placeholder[
                        :, layer_drop, :, : hidden_states[layer_drop + 1].shape[-1]
                    ]
                    * (1 - gates.unsqueeze(-1))
                ]
                + [None] * (len(hidden_states) - layer_drop + 1 - 1)
            ),
        )

        return (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        )

    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        query_ids, input_ids, mask, labels = batch

        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self.forward_explainer(query_ids, input_ids, mask)

        loss_c = (
            torch.distributions.kl_divergence(
                torch.distributions.Bernoulli(logits=logits_orig),
                torch.distributions.Bernoulli(logits=logits),
            )
            - self.hparams.eps
        ).mean(-1)

        loss_g = ((expected_L0 * mask).sum(-1) / mask.sum(-1)).mean(-1)

        loss = self.alpha[layer_drop] * loss_c + loss_g

        acc = ((logits > 0).float() == labels).float().mean(-1) * 100

        l0 = ((expected_L0.exp() * mask).sum(-1) / mask.sum(-1)).mean(-1)

        outputs_dict = {
            "acc": acc,
            "alpha": self.alpha[layer_drop],
            "loss_g": loss_g,
            "loss_c": loss_c,
            "l0": l0,
        }

        outputs_dict = {
            k if self.training else "val_" + k: v for k, v in outputs_dict.items()
        }

        return {"loss": loss, **outputs_dict, "progress_bar": outputs_dict}

    def validation_epoch_end(self, outputs):

        loss_c = sum(e["val_loss_c"] for e in outputs) / len(outputs) + self.hparams.eps
        loss_g = sum(e["val_loss_g"] for e in outputs) / len(outputs)

        acc = sum(e["val_acc"] for e in outputs) / len(outputs)
        l0 = sum(e["val_l0"] for e in outputs) / len(outputs)

        outputs_dict = {
            "val_loss_c": loss_c,
            "val_loss_g": loss_g,
            "val_alpha": self.alpha.mean(-1),
            "val_acc": acc,
            "val_l0": l0,
        }

        outputs_dict = {
            "val_loss": l0,
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        optimizers = [
            LookaheadRMSprop(
                params=list(self.gate.parameters()) + [self.placeholder],
                lr=self.hparams.learning_rate,
                centered=True,
            ),
            LookaheadRMSprop(params=[self.alpha], lr=self.hparams.learning_rate_alpha,),
        ]

        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 200),
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

        for g in optimizer.param_groups:
            for p in g["params"]:
                p.grad = None


@torch.distributions.kl.register_kl(
    torch.distributions.Bernoulli, torch.distributions.Bernoulli
)
def _kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (torch.log(p.probs + 1e-5) - torch.log(q.probs + 1e-5))
    t2 = (1 - p.probs) * (torch.log1p(-p.probs + 1e-5) - torch.log1p(-q.probs + 1e-5))
    return t1 + t2
