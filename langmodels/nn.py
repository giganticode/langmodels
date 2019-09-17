from typing import List, Tuple

import torch
from fastai.text import SequentialRNN

TORCH_LONG_MIN_VAL = -2 ** 63


def to_test_mode(model: SequentialRNN) -> None:
    # Set batch size to 1
    model[0].bs = 1
    # Turn off dropout
    model.eval()
    # Reset hidden state
    model.reset()


def save_hidden_states(model: SequentialRNN) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(hl[0].clone(), hl[1].clone()) for hl in model[0].hidden]


def get_last_layer_activations(model: SequentialRNN, input: torch.FloatTensor) -> torch.FloatTensor:
    tensor_rank = len(input.size())
    if tensor_rank != 2:
        if tensor_rank == 0:
            error_msg = 'Your input tensor is a scalar. If you want to use batch size = 1 and feed only one token to the model, pass input[None, None] to this method.'
        elif tensor_rank == 1:
            error_msg = 'You input tensor has rank 1. If you were intending to use batch size = 1, please pass input[None, :] to this method. If you however were intending to use multiple batches but feed only one element to the model pass input[:, None]'
        else:
            error_msg = f'Your tensor has rank {tensor_rank}.'

        raise ValueError(f'This method accepts tensors of rank 2. {error_msg}')

    if input.nelement() == 0:
        return None

    last_layer_activations, *_ = model(input)
    last_token_predictions = last_layer_activations[:, -1]
    return last_token_predictions
