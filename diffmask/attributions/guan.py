import torch
from tqdm.auto import trange
from ..utils.getter_setter import (
    bert_getter,
    bert_setter,
    gru_getter,
    gru_setter,
)


def guan_explainer(
    model,
    inputs_dict,
    getter,
    setter,
    s_fn,
    loss_l2_fn,
    loss_h_fn,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    with torch.no_grad():
        outputs, hidden_states = getter(model, inputs_dict)
        s = s_fn(outputs, hidden_states)

    sigma = torch.full(
        hidden_states[hidden_state_idx].shape[:-1],
        -5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    optimizer = torch.optim.RMSprop([sigma], lr=lr, centered=True)

    t = trange(steps)
    for _ in t:
        optimizer.zero_grad()

        eps = torch.distributions.Normal(
            loc=torch.zeros_like(sigma), scale=torch.nn.functional.softplus(sigma),
        )

        noise = eps.rsample((hidden_states[hidden_state_idx].shape[-1],)).permute(
            list(range(1, len(hidden_states[hidden_state_idx].shape))) + [0]
        )

        s_pred = s_fn(
            *setter(
                model,
                inputs_dict,
                hidden_states=[None] * hidden_state_idx
                + [hidden_states[hidden_state_idx] + noise]
                + [None] * (len(hidden_states) - hidden_state_idx - 1),
            )
        )

        loss_l2 = loss_l2_fn(
            [(s_i - s_pred_i) ** 2 for s_i, s_pred_i in zip(s, s_pred)], inputs_dict
        )
        loss_h = loss_h_fn(eps.entropy(), inputs_dict)

        loss = loss_l2 - la * loss_h

        loss.backward()
        optimizer.step()

        t.set_postfix(
            loss="{:.2f}".format(loss.item()),
            loss_l2="{:.2f}".format(loss_l2.item()),
            loss_h="{:.2f}".format(-loss_h.item()),
            refresh=False,
        )

    return torch.nn.functional.softplus(sigma).detach()


def sst_bert_guan_explainer(
    model, inputs_dict, hidden_state_idx=0, steps=10, lr=1e-1, la=10,
):
    return guan_explainer(
        model=model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "labels": inputs_dict["labels"],
        },
        getter=bert_getter,
        setter=bert_setter,
        s_fn=lambda outputs, hidden_states: outputs[1:],
        loss_l2_fn=lambda s, inputs_dict: sum(s_i.sum(-1).mean(-1) for s_i in s),
        loss_h_fn=lambda h, inputs_dict: (h * inputs_dict["attention_mask"])
        .sum(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )


def sst_gru_guan_explainer(
    model, inputs_dict, hidden_state_idx=0, steps=10, lr=1e-1, la=10,
):
    device = next(model.parameters()).device
    model.train()
    output = guan_explainer(
        model=model.net,
        inputs_dict=inputs_dict,
        getter=gru_getter,
        setter=gru_setter,
        s_fn=lambda outputs, hidden_states: outputs,
        loss_l2_fn=lambda s, inputs_dict: sum(s_i.sum(-1).mean(-1) for s_i in s),
        loss_h_fn=lambda h, inputs_dict: (h * inputs_dict["mask"]).sum(-1).mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
    model.eval()
    return output


def squad_bert_guan_explainer(
    model, inputs_dict, hidden_state_idx=0, steps=10, lr=1e-1, la=10,
):
    return guan_explainer(
        model=model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "token_type_ids": inputs_dict["token_type_ids"],
            "start_positions": inputs_dict["start_positions"],
            "end_positions": inputs_dict["end_positions"],
        },
        getter=bert_getter,
        setter=bert_setter,
        s_fn=lambda outputs, hidden_states: outputs[1:],
        loss_l2_fn=lambda s, inputs_dict: sum(
            (s_i * inputs_dict["attention_mask"]).sum(-1).mean(-1) for s_i in s
        ),
        loss_h_fn=lambda h, inputs_dict: (h * inputs_dict["attention_mask"])
        .sum(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
