import torch
from transformers import BertForSequenceClassification
from collections import defaultdict


def bert_getter(model, inputs_dict, forward_fn=None):

    hidden_states_ = []

    def get_hook(i):
        def hook(module, inputs, outputs=None):
            if i == 0:
                hidden_states_.append(outputs)
            elif 1 <= i <= len(model.bert.encoder.layer):
                hidden_states_.append(inputs[0])
            elif i == len(model.bert.encoder.layer) + 1:
                hidden_states_.append(outputs[0])

        return hook

    handles = (
        [model.bert.embeddings.word_embeddings.register_forward_hook(get_hook(0))]
        + [
            layer.register_forward_pre_hook(get_hook(i + 1))
            for i, layer in enumerate(model.bert.encoder.layer)
        ]
        + [
            model.bert.encoder.layer[-1].register_forward_hook(
                get_hook(len(model.bert.encoder.layer) + 1)
            )
        ]
    )

    try:
        if forward_fn is None:
            outputs = model(**inputs_dict)
        else:
            outputs = forward_fn(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def bert_setter(model, inputs_dict, hidden_states, forward_fn=None):

    hidden_states_ = []

    def get_hook(i):
        def hook(module, inputs, outputs=None):
            if i == 0:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return hidden_states[i]
                else:
                    hidden_states_.append(outputs)

            elif 1 <= i <= len(model.bert.encoder.layer):
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + inputs[1:]
                else:
                    hidden_states_.append(inputs[0])

            elif i == len(model.bert.encoder.layer) + 1:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + outputs[1:]
                else:
                    hidden_states_.append(outputs[0])

        return hook

    handles = (
        [model.bert.embeddings.word_embeddings.register_forward_hook(get_hook(0))]
        + [
            layer.register_forward_pre_hook(get_hook(i + 1))
            for i, layer in enumerate(model.bert.encoder.layer)
        ]
        + [
            model.bert.encoder.layer[-1].register_forward_hook(
                get_hook(len(model.bert.encoder.layer) + 1)
            )
        ]
    )

    try:
        if forward_fn is None:
            outputs = model(**inputs_dict)
        else:
            outputs = forward_fn(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


# def bert_setter(model, inputs_dict, hidden_states, forward_fn=None):

#     hidden_states_ = []

#     def get_hook(i):
#         def hook(module, inputs, outputs=None):
#             if i == 0:
#                 if hidden_states[i] is not None:
#                     hidden_states_.append(hidden_states[i])
#                     return hidden_states[i]
#                 else:
#                     hidden_states_.append(outputs)

#             elif 1 <= i <= len(model.bert.encoder.layer):
#                 if hidden_states[i] is not None:
#                     hidden_states_.append(hidden_states[i])
#                     return hidden_states[i] + inputs[2:]
#                 else:
#                     hidden_states_.append(inputs[1])

#             elif i == len(model.bert.encoder.layer) + 1:
#                 if hidden_states[i] is not None:
#                     hidden_states_.append(hidden_states[i])
#                     return (hidden_states[i],) + outputs[1:]
#                 else:
#                     hidden_states_.append(outputs[0])

#         return hook

#     handles = (
#         [model.bert.embeddings.word_embeddings.register_forward_hook(get_hook(0))]
#         + [
#             layer.register_forward_pre_hook(get_hook(i + 1))
#             for i, layer in enumerate(model.bert.encoder.layer)
#         ]
#         + [
#             model.bert.encoder.layer[-1].register_forward_hook(
#                 get_hook(len(model.bert.encoder.layer) + 1)
#             )
#         ]
#     )

#     try:
#         if forward_fn is None:
#             outputs = model(**inputs_dict)
#         else:
#             outputs = forward_fn(**inputs_dict)
#     finally:
#         for handle in handles:
#             handle.remove()

#     return outputs, tuple(hidden_states_)


def gru_getter(model, inputs_dict):

    hidden_states_ = []

    def get_hook():
        def hook(module, inputs, outputs):
            hidden_states_.append(outputs)

        return hook

    handles = [model.emb.register_forward_hook(get_hook())]

    try:
        outputs = model(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def gru_setter(model, inputs_dict, hidden_states):

    hidden_states_ = []

    def get_hook():
        def hook(module, inputs, outputs):
            if hidden_states[0] is not None:
                hidden_states_.append(hidden_states[0])
                return hidden_states[0]
            else:
                hidden_states_.append(outputs)

        return hook

    handles = [model.emb.register_forward_hook(get_hook())]

    try:
        outputs = model(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def toy_getter(model, inputs_dict):

    hidden_states_ = []
    handles = [
        model.embedding_query.register_forward_hook(
            lambda module, inputs, outputs: hidden_states_.append(outputs)
        ),
        model.embedding_input.register_forward_hook(
            lambda module, inputs, outputs: hidden_states_.append(outputs)
        ),
        model.encoder.register_forward_hook(
            lambda module, inputs, outputs: hidden_states_.append(outputs)
        ),
    ]

    outputs = model(**inputs_dict)
    for handle in handles:
        handle.remove()

    return outputs, tuple(hidden_states_)


def toy_setter(model, inputs_dict, hidden_states):

    handles = [
        model.embedding_query.register_forward_hook(
            lambda module, inputs, outputs: hidden_states[0]
            if hidden_states[0] is not None
            else None
        ),
        model.embedding_input.register_forward_hook(
            lambda module, inputs, outputs: hidden_states[1]
            if hidden_states[1] is not None
            else None
        ),
        model.encoder.register_forward_hook(
            lambda module, inputs, outputs: hidden_states[2]
            if hidden_states[2] is not None
            else None
        ),
    ]

    outputs = model(**inputs_dict)
    for handle in handles:
        handle.remove()

    return outputs, None


def test_bert_getter():

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").cuda()

    inputs_dict = {
        "input_ids": torch.tensor(
            [[101, 1037, 4010, 1010, 6057, 1010, 11973, 2143, 1012, 102, 0, 0, 0,]]
        ).cuda(),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,]]
        ).cuda(),
        "labels": torch.tensor([1]).cuda(),
    }

    model.bert.encoder.output_hidden_states = True

    outputs_orig = model(**inputs_dict)

    outputs_orig = (
        outputs_orig[0],
        outputs_orig[1],
        (model.bert.embeddings.word_embeddings(inputs_dict["input_ids"]),)
        + outputs_orig[2],
    )

    model.bert.encoder.output_hidden_states = False

    outputs = bert_getter(model, inputs_dict)

    assert (outputs_orig[0] == outputs[0][0]).all()

    assert (outputs_orig[1] == outputs[0][1]).all()

    assert all((a == b).all() for a, b in zip(outputs_orig[2], outputs[1]))


def test_bert_setter():

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").cuda()

    inputs_dict = {
        "input_ids": torch.tensor(
            [[101, 1037, 4010, 1010, 6057, 1010, 11973, 2143, 1012, 102, 0, 0, 0,]]
        ).cuda(),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,]]
        ).cuda(),
        "labels": torch.tensor([1]).cuda(),
    }

    outputs, hidden_states = bert_getter(model, inputs_dict)

    for i, h in enumerate(hidden_states):
        outputs_, hidden_states_ = bert_setter(
            model,
            inputs_dict,
            [None] * i + [h * 0] + [None] * (len(hidden_states) - 1 - i),
        )
        assert all((a != b).all() for a, b in zip(outputs, outputs_))
