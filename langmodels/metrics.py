from torch import Tensor

def _find_sorted_array_position(tensor: Tensor, values_tensor: Tensor) -> Tensor:
    dim0, dim1 = tensor.shape
    expanded_values_tensor = values_tensor.resize_((dim0, 1)).expand(dim0, dim1)
    position_of_value = torch.sum((tensor > expanded_values_tensor).long(), 1)
    return position_of_value.add(1)


def mrr(preds: Tensor, targs: Tensor) -> Tensor:
    """
    E.g.:
    preds = torch.tensor([[0.2, 0.55, 0.25], [0.005, 0.005, 0.99]])
    targs = torch.tensor([1, 2])
    -> 1.0
    """
    pred_values = preds.gather(1, targs.view(-1, 1))
    guessed_positions = _find_sorted_array_position(preds, pred_values).float()
    reciprocal = torch.reciprocal(guessed_positions)
    return torch.mean(reciprocal)