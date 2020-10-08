import torch
from tqdm.auto import tqdm, trange
from ..utils.getter_setter import (
    bert_getter,
    bert_setter,
    gru_getter,
    gru_setter,
    toy_getter,
    toy_setter,
)


def schulz_explainer(
    model,
    inputs_dict,
    getter,
    setter,
    q_z_loc,
    q_z_scale,
    loss_fn,
    loss_kl_fn,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):

    with torch.no_grad():
        _, hidden_states = getter(model, inputs_dict)

    alpha = torch.full(
        hidden_states[hidden_state_idx].shape[:-1],
        5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    optimizer = torch.optim.RMSprop([alpha], lr=lr, centered=True)

    t = trange(steps)
    for _ in t:
        optimizer.zero_grad()
        gates = alpha.sigmoid()

        p_z_r = torch.distributions.Normal(
            loc=gates.unsqueeze(-1) * hidden_states[hidden_state_idx]
            + (1 - gates).unsqueeze(-1) * q_z_loc,
            scale=(q_z_scale + 1e-8) * (1 - gates).unsqueeze(-1),
        )

        q_z = torch.distributions.Normal(loc=q_z_loc, scale=(q_z_scale + 1e-8),)

        loss_model = loss_fn(
            *setter(
                model,
                inputs_dict,
                hidden_states=[None] * hidden_state_idx
                + [p_z_r.rsample()]
                + [None] * (len(hidden_states) - hidden_state_idx - 1),
            ),
            inputs_dict,
        )

        loss_kl = loss_kl_fn(
            torch.distributions.kl_divergence(p_z_r, q_z).mean(-1), inputs_dict
        )

        loss = loss_model + la * loss_kl

        loss.backward()
        optimizer.step()

        t.set_postfix(
            loss="{:.2f}".format(loss.item()),
            loss_model="{:.2f}".format(loss_model.item()),
            loss_kl="{:.2f}".format(loss_kl.item()),
            refresh=False,
        )

    attributions = alpha.sigmoid().detach()

    return attributions


def bert_hidden_states_statistics(model, input_only=True):

    with torch.no_grad():
        all_hidden_states = []
        for batch in tqdm(model.train_dataloader()):
            batch = tuple(e.to(next(model.parameters()).device) for e in batch)
            if input_only:
                hidden_states = [model.net.bert.embeddings.word_embeddings(batch[0])]
            else:
                _, hidden_states = bert_getter(
                    model.net,
                    {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        **({"token_type_ids": batch[2]} if len(batch) == 5 else {}),
                    },
                )
            all_hidden_states.append(torch.stack(hidden_states).cpu())

        all_q_z_loc = sum([e.sum(1) for e in all_hidden_states]) / sum(
            [e.shape[1] for e in all_hidden_states]
        )
        all_q_z_scale = (
            sum(((all_q_z_loc.unsqueeze(1) - e) ** 2).sum(1) for e in all_hidden_states)
            / sum([e.shape[1] for e in all_hidden_states])
        ).sqrt()

    return all_q_z_loc, all_q_z_scale


def sst_gru_hidden_states_statistics(model):

    with torch.no_grad():
        all_hidden_states = []
        for batch in tqdm(model.train_dataloader()):
            batch = tuple(e.to(next(model.parameters()).device) for e in batch)
            _, hidden_states = gru_getter(
                model.net, {"input_ids": batch[0], "mask": batch[1],}
            )
            all_hidden_states.append(torch.stack(hidden_states).cpu())

        all_hidden_states = torch.cat(all_hidden_states, 1)
        all_q_z_loc = all_hidden_states.mean(1)
        all_q_z_scale = all_hidden_states.std(1)

        return all_q_z_loc, all_q_z_scale


def toy_hidden_states_statistics(model):

    all_hidden_states = [[], [], []]
    for batch in tqdm(model.train_dataloader()):
        batch = tuple(e.to(next(model.parameters()).device) for e in batch)
        _, hidden_states = toy_getter(
            model, {"query_ids": batch[0], "input_ids": batch[1], "mask": batch[2],}
        )
        all_hidden_states[0].append(hidden_states[0].cpu())
        all_hidden_states[1].append(hidden_states[1].cpu())
        all_hidden_states[2].append(hidden_states[2].cpu())

    all_hidden_states = [torch.cat(e, 0) for e in all_hidden_states]
    all_q_z_loc = [e.mean(0) for e in all_hidden_states]
    all_q_z_scale = [e.std(0) for e in all_hidden_states]

    return all_q_z_loc, all_q_z_scale


def sst_bert_schulz_explainer(
    model,
    inputs_dict,
    all_q_z_loc,
    all_q_z_scale,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    device = next(model.parameters()).device
    return schulz_explainer(
        model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "labels": inputs_dict["labels"],
        },
        getter=bert_getter,
        setter=bert_setter,
        q_z_loc=all_q_z_loc[hidden_state_idx].unsqueeze(0).to(device),
        q_z_scale=all_q_z_scale[hidden_state_idx].unsqueeze(0).to(device),
        loss_fn=lambda outputs, hidden_states, inputs_dict: outputs[0],
        loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["attention_mask"])
        .mean(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )


def sst_gru_schulz_explainer(
    model,
    inputs_dict,
    all_q_z_loc,
    all_q_z_scale,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    device = next(model.parameters()).device
    model.train()
    output = schulz_explainer(
        model.net,
        inputs_dict=inputs_dict,
        getter=gru_getter,
        setter=gru_setter,
        q_z_loc=all_q_z_loc[hidden_state_idx].unsqueeze(0).to(device),
        q_z_scale=all_q_z_scale[hidden_state_idx].unsqueeze(0).to(device),
        loss_fn=lambda outputs, hidden_states, inputs_dict: model.training_step_end(
            outputs + (inputs_dict["labels"],)
        )["loss"],
        loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["mask"]).mean(-1).mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
    model.eval()
    return output


def squad_bert_schulz_explainer(
    model,
    inputs_dict,
    all_q_z_loc,
    all_q_z_scale,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    device = next(model.parameters()).device
    return schulz_explainer(
        model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "token_type_ids": inputs_dict["token_type_ids"],
            "start_positions": inputs_dict["start_positions"],
            "end_positions": inputs_dict["end_positions"],
        },
        getter=bert_getter,
        setter=bert_setter,
        q_z_loc=all_q_z_loc[hidden_state_idx].unsqueeze(0).to(device),
        q_z_scale=all_q_z_scale[hidden_state_idx].unsqueeze(0).to(device),
        loss_fn=lambda outputs, hidden_states, inputs_dict: outputs[0],
        loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["attention_mask"])
        .mean(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
