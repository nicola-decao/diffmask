import numpy as np
import torch
from .distributions import RectifiedStreched, BinaryConcrete


class MLPGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.utils.weight_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.Tanh(),
            torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, 1, bias=bias)),
        )
        if bias:
            self.f[-1].bias.data[:] = 5.0

    def forward(self, *args):
        return self.f(torch.cat(args, -1))


class MLPMaxGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, max_activation=10, bias=True):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.utils.weight_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.Tanh(),
            torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, 1, bias=bias)),
            torch.nn.Tanh(),
        )
        self.bias = torch.nn.Parameter(torch.tensor(5.0))
        self.max_activation = max_activation

    def forward(self, *args):
        return self.f(torch.cat(args, -1)) * self.max_activation + self.bias


class DiffMaskGateInput(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gate_fn: torch.nn.Module = MLPMaxGate,
        gate_bias: bool = True,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.g_hat = torch.nn.ModuleList(
            [
                gate_fn(hidden_size * 2, hidden_attention, bias=gate_bias)
                for _ in range(num_hidden_layers)
            ]
        )

        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(1, max_position_embeddings, hidden_size,)
                )
                if init_vector is None
                else init_vector.view(1, 1, hidden_size).repeat(
                    1, max_position_embeddings, 1
                )
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((1, 1, hidden_size,)),
            )

    def forward(self, hidden_states, mask, layer_pred):

        logits = torch.cat(
            [
                self.g_hat[i](hidden_states[0], hidden_states[i])
                for i in range(
                    (layer_pred + 1) if layer_pred is not None else len(hidden_states)
                )
            ],
            -1,
        )

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )

        gates_full = dist.rsample().cumprod(-1)
        expected_L0_full = dist.log_expected_L0().cumsum(-1)

        gates = gates_full[..., -1]
        expected_L0 = expected_L0_full[..., -1]

        return (
            hidden_states[0] * gates.unsqueeze(-1)
            + self.placeholder[:, : hidden_states[0].shape[-2],]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )


class DiffMaskGateHidden(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gate_fn: torch.nn.Module = MLPMaxGate,
        gate_bias: bool = True,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.g_hat = torch.nn.ModuleList(
            [
                gate_fn(hidden_size, hidden_attention, bias=gate_bias)
                for _ in range(num_hidden_layers)
            ]
        )

        if placeholder:
            self.placeholder = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.nn.init.xavier_normal_(
                            torch.empty(1, max_position_embeddings, hidden_size,)
                        )
                        if init_vector is None
                        else init_vector.view(1, 1, hidden_size).repeat(
                            1, max_position_embeddings, 1
                        )
                    )
                    for _ in range(num_hidden_layers)
                ]
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((num_hidden_layers, 1, 1, hidden_size,)),
            )

    def forward(self, hidden_states, mask, layer_pred):

        if layer_pred is not None:
            logits = self.g_hat[layer_pred](hidden_states[layer_pred])
        else:
            logits = torch.cat(
                [self.g_hat[i](hidden_states[i]) for i in range(len(hidden_states))], -1
            )

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )

        gates_full = dist.rsample()
        expected_L0_full = dist.log_expected_L0()

        gates = gates_full if layer_pred is not None else gates_full[..., :1]
        expected_L0 = (
            expected_L0_full if layer_pred is not None else expected_L0_full[..., :1]
        )

        return (
            hidden_states[layer_pred if layer_pred is not None else 0] * gates
            + self.placeholder[layer_pred if layer_pred is not None else 0][
                :,
                : hidden_states[layer_pred if layer_pred is not None else 0].shape[-2],
            ]
            * (1 - gates),
            gates.squeeze(-1),
            expected_L0.squeeze(-1),
            gates_full,
            expected_L0_full,
        )


class PerSampleGate(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        batch_size: int = 1,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.logits = torch.nn.Parameter(
            torch.full((batch_size, max_position_embeddings, num_hidden_layers), 5.0)
        )

        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        batch_size,
                        num_hidden_layers,
                        max_position_embeddings,
                        hidden_size,
                    )
                )
                if init_vector is None
                else init_vector.view(1, num_hidden_layers, 1, hidden_size).repeat(
                    batch_size, 1, max_position_embeddings, 1
                )
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((1, num_hidden_layers, 1, hidden_size))
            )


class PerSampleDiffMaskGate(PerSampleGate):
    def forward(self, hidden_states, mask, layer_pred):

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(self.logits, 0.2), self.logits),
            l=-0.2,
            r=1.0,
        )

        gates_full = dist.rsample()
        expected_L0_full = dist.log_expected_L0()

        gates = gates_full[..., layer_pred]
        expected_L0 = expected_L0_full[..., layer_pred]

        return (
            hidden_states[layer_pred] * gates.unsqueeze(-1)
            + self.placeholder[:, layer_pred, : hidden_states[layer_pred].shape[-2]]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )


class PerSampleREINFORCEGate(PerSampleGate):
    def forward(self, hidden_states, mask, layer_pred):

        dist = torch.distributions.Bernoulli(logits=self.logits)

        gates_full = dist.sample()
        expected_L0_full = dist.log_prob(1.0)

        gates = gates_full[..., layer_pred]
        expected_L0 = expected_L0_full[..., layer_pred]

        return (
            hidden_states[layer_pred] * gates.unsqueeze(-1)
            + self.placeholder[:, layer_pred, : hidden_states[layer_pred].shape[-2]]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )
