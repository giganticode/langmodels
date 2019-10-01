import torch
from torch import Tensor


def _find_sorted_array_position(tensor: Tensor, values_tensor: Tensor) -> Tensor:
    dim0, dim1, dim2 = tensor.shape
    expanded_values_tensor = values_tensor.resize_((dim0, dim1, 1)).expand(dim0, dim1, dim2)
    position_of_value = torch.sum((tensor > expanded_values_tensor).long(), -1)
    return position_of_value.add(1)


def mrr(preds: Tensor, targs: Tensor) -> Tensor:
    """
    >>> preds = torch.tensor([[[0.2, 0.55, 0.25], [0.005, 0.005, 0.99]]])
    >>> targs = torch.tensor([[1, 2]])
    >>> mrr(preds, targs)
    tensor(1.)

    >>> preds = torch.tensor([[[0.2, 0.55, 0.25], [0.006, 0.004, 0.99]]])
    >>> targs = torch.tensor([[0, 0]])
    >>> mrr(preds, targs)
    tensor(0.4167)
    """
    pred_values = preds.gather(-1, targs.unsqueeze(-1))
    guessed_positions = _find_sorted_array_position(preds, pred_values).float()
    reciprocal = torch.reciprocal(guessed_positions)
    return torch.mean(reciprocal)
