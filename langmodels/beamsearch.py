from typing import Tuple, Callable

import torch
from torch import FloatTensor, LongTensor, Tensor
from dataclasses import dataclass
from fastai.text import SequentialRNN
from torch.nn.functional import log_softmax

from langmodels.nn import get_last_layer_activations, save_hidden_states, TORCH_LONG_MIN_VAL
from langmodels.cuda_util import get_device

DEVICE = get_device()


def _get_topk_predictions(model: SequentialRNN, context: FloatTensor, top_k: int) -> Tuple[FloatTensor, LongTensor]:
    last_token_activations = get_last_layer_activations(model, context)
    predictions = log_softmax(last_token_activations[:, -1], dim=-1)  # TODO log_softmax not really needed
    return predictions.topk(top_k, dim=-1)


def _topk_are_full_tokens(full_token_flags_sorted: torch.Tensor, top_k: int) -> bool:
    return full_token_flags_sorted.size(0) >= top_k and full_token_flags_sorted[top_k-1].item() == (top_k - 1)


@dataclass
class FlattenedCandidateList(object):
    """
    >>> candidate_list = FlattenedCandidateList.of(LongTensor([[1, 2, 3], [4, 5, 6]]), FloatTensor([0.3, 0.1]))
    >>> candidate_list
    FlattenedCandidateList(subtokens=tensor([[1, 2, 3],
            [4, 5, 6]]), scores=tensor([0.3000, 0.1000]), hidden_indices=tensor([-9223372036854775808, -9223372036854775808]))
    >>> candidate_list = candidate_list.add(candidate_list)
    >>> candidate_list
    FlattenedCandidateList(subtokens=tensor([[1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [4, 5, 6]]), scores=tensor([0.3000, 0.1000, 0.3000, 0.1000]), hidden_indices=tensor([-9223372036854775808, -9223372036854775808, -9223372036854775808,
            -9223372036854775808]))
    >>> candidate_list.sort()
    FlattenedCandidateList(subtokens=tensor([[4, 5, 6],
            [4, 5, 6],
            [1, 2, 3],
            [1, 2, 3]]), scores=tensor([0.1000, 0.1000, 0.3000, 0.3000]), hidden_indices=tensor([-9223372036854775808, -9223372036854775808, -9223372036854775808,
            -9223372036854775808]))
    >>> candidate_list.size()
    4
    >>> candidate_list.sub_list(torch.LongTensor([3, 2, 1]))
    FlattenedCandidateList(subtokens=tensor([[4, 5, 6],
            [1, 2, 3],
            [4, 5, 6]]), scores=tensor([0.1000, 0.3000, 0.1000]), hidden_indices=tensor([-9223372036854775808, -9223372036854775808, -9223372036854775808]))
    >>> candidate_list.add_empty_subword_to_all()
    >>> candidate_list
    FlattenedCandidateList(subtokens=tensor([[                   1,                    2,                    3,
             -9223372036854775808],
            [                   4,                    5,                    6,
             -9223372036854775808],
            [                   1,                    2,                    3,
             -9223372036854775808],
            [                   4,                    5,                    6,
             -9223372036854775808]]), scores=tensor([0.3000, 0.1000, 0.3000, 0.1000]), hidden_indices=tensor([-9223372036854775808, -9223372036854775808, -9223372036854775808,
            -9223372036854775808]))
    """
    subtokens: Tensor
    scores: Tensor
    hidden_indices: Tensor

    def __post_init__(self):
        assert len(self.subtokens.size()) == 2 and isinstance(self.subtokens, LongTensor)
        assert len(self.scores.size()) == 1 and isinstance(self.scores, FloatTensor)
        assert len(self.hidden_indices.size()) == 1 and isinstance(self.hidden_indices, LongTensor)

        assert self.subtokens.size(0) == self.scores.size(0) == self.hidden_indices.size(0)

    def size(self):
        return self.subtokens.size(0)

    @staticmethod
    def of(subtokens: Tensor, scores: Tensor) -> 'FlattenedCandidateList':
        n_full_tokens = scores.size(0)
        assert n_full_tokens == subtokens.size(0)

        best_tokens = FlattenedCandidateList(subtokens=subtokens, scores=scores,
                                             hidden_indices=torch.full((n_full_tokens,), fill_value=TORCH_LONG_MIN_VAL, dtype=torch.long, device=DEVICE))
        return best_tokens

    @staticmethod
    def empty():
        return FlattenedCandidateList.of(
            subtokens=torch.empty((0, 0), dtype=torch.long, device=DEVICE),
            scores=torch.empty(0, dtype=torch.float, device=DEVICE)
        )

    def add(self, candidates: 'FlattenedCandidateList') -> 'FlattenedCandidateList':
        return FlattenedCandidateList(subtokens=torch.cat([self.subtokens, candidates.subtokens], dim=0),
                                      scores=torch.cat([self.scores, candidates.scores], dim=0),
                                      hidden_indices=torch.cat([self.hidden_indices, candidates.hidden_indices], dim=0))

    def sort(self) -> 'FlattenedCandidateList':
        idx_in_score_asc_order = self.scores.argsort()
        return FlattenedCandidateList(
            subtokens= self.subtokens[idx_in_score_asc_order],
            scores= self.scores[idx_in_score_asc_order],
            hidden_indices=self.hidden_indices[idx_in_score_asc_order]
        )

    def sub_list(self, indices: Tensor) -> 'FlattenedCandidateList':
        return FlattenedCandidateList(
            subtokens=self.subtokens[indices],
            scores=self.scores[indices],
            hidden_indices=self.hidden_indices[indices],
        )

    @staticmethod
    def single_empty():
        return FlattenedCandidateList.of(subtokens=torch.empty(1, 0, device=DEVICE).long(), scores=torch.zeros(1, device=DEVICE).float())

    def add_empty_subword_to_all(self) -> None:
        self.subtokens = torch.cat([self.subtokens,
                   torch.full((self.size(), 1), fill_value=TORCH_LONG_MIN_VAL, dtype=torch.long, device=DEVICE)], dim=-1)


def _expand_with_new_candidates(model: SequentialRNN, context: LongTensor, n_predictions: int,
                                current_candidates: FlattenedCandidateList) -> FlattenedCandidateList:
    assert len(context.size()) == 2

    loss, num_tokens = _get_topk_predictions(model, context, n_predictions)
    batch_size = context.size(0)

    return FlattenedCandidateList(scores= (-loss + current_candidates.scores[:, None]).view(-1),
                                  hidden_indices= torch.arange(0, batch_size, device=DEVICE)[:, None].expand(batch_size, n_predictions).contiguous().view(-1),
                                  subtokens= torch.cat([current_candidates.subtokens.repeat(n_predictions, 1), num_tokens.view(-1)[:, None]], dim=-1))


CompleteTokenPredicate = Callable[[Tensor], Tuple[Tensor, Tensor]]


def beam_search(model: SequentialRNN, context: torch.LongTensor, complete_token_predicate: CompleteTokenPredicate,
                top_k: int, beam_size: int) -> Tuple[Tensor, Tensor]:
    if len(context.size()) != 1:
        raise ValueError("The rank of context tensor should be one. Beam search is only possible with batch size = 1.")
    if top_k > beam_size:
        raise ValueError(f"N suggestions ({top_k}) cannot be more than the beam size ({beam_size})")

    context = context.unsqueeze(dim=0)
    checkpoint = save_hidden_states(model)
    pending_candidates_sorted = FlattenedCandidateList.single_empty()
    best_tokens = FlattenedCandidateList.empty()
    ready_candidate_indices = torch.empty(0)
    with torch.no_grad():
        while not _topk_are_full_tokens(ready_candidate_indices, top_k):
            new_candidates = _expand_with_new_candidates(model, context, beam_size, pending_candidates_sorted)
            best_tokens.add_empty_subword_to_all()
            all_candidates = best_tokens.add(new_candidates)
            best_candidates_sorted = all_candidates.sort().sub_list(torch.arange(beam_size))
            ready_candidate_indices, pending_candidate_idxs = complete_token_predicate(best_candidates_sorted.subtokens)
            best_tokens = best_candidates_sorted.sub_list(ready_candidate_indices)
            pending_candidates_sorted = best_candidates_sorted.sub_list(pending_candidate_idxs)
            model[0].select_hidden(pending_candidates_sorted.hidden_indices)
            context = pending_candidates_sorted.subtokens[:, -1:]

    model[0].hidden = checkpoint
    model[0].bs = checkpoint[0][0].size(0)

    return best_tokens.subtokens[:top_k], best_tokens.scores[:top_k]
