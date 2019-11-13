from typing import Tuple, List, Dict

import torch
from fastai.text import SequentialRNN
from torch.nn.functional import log_softmax

from langmodels.nn import get_last_layer_activations, save_hidden_states, TORCH_LONG_MIN_VAL
from langmodels.cuda_util import get_device

DEVICE = get_device()


def get_topk_predictions(model: SequentialRNN, context: torch.FloatTensor, top_k: int) -> Tuple[
    torch.FloatTensor, torch.LongTensor]:
    last_token_activations = get_last_layer_activations(model, context)
    predictions = log_softmax(last_token_activations[:, -1], dim=-1)  # TODO log_softmax not really needed
    return predictions.topk(top_k, dim=-1)


def topk_are_full_tokens(full_tokens_bitmap: torch.Tensor, top_k: int) -> bool:
    return full_tokens_bitmap[:top_k].sum() == top_k


Tensors = Dict[str, torch.Tensor]


def get_new_candidates(model: SequentialRNN, context: torch.LongTensor, n_predictions: int,
                       pending_candidates: Tensors) -> Tensors:
    assert len(context.size()) == 2

    loss, num_tokens = get_topk_predictions(model, context, n_predictions)
    batch_size = context.size(0)

    new_candidates = {'scores': (-loss + pending_candidates['scores'][:, None]).view(-1),
                      'num_tokens': num_tokens.view(-1),
                      'hidden_indices': torch.arange(0, batch_size, device=DEVICE)[:, None] \
                          .expand(batch_size, n_predictions).contiguous().view(-1),
                      'tokens': pending_candidates['tokens'].repeat(n_predictions, 1)}
    return new_candidates


def update_ready_candidates(best_subtokens: torch.LongTensor, best_scores: torch.FloatTensor) -> Tensors:
    ready_candidates = {'tokens': best_subtokens, 'scores': best_scores}
    n_full_tokens = best_scores.size(0)
    ready_candidates['num_tokens'] = torch.full((n_full_tokens,), fill_value=TORCH_LONG_MIN_VAL, dtype=torch.long, device=DEVICE)
    ready_candidates['hidden_indices'] = torch.full((n_full_tokens,), fill_value=TORCH_LONG_MIN_VAL, dtype=torch.long, device=DEVICE)
    return ready_candidates


def beam_search(model: SequentialRNN, context: torch.LongTensor, first_nonterm_token: int, top_k: int,
                beam_size: int) -> List[Tuple]:
    if len(context.size()) != 1:
        raise ValueError("The rank of context tensor should be one. Beam search is only possible with batch size = 1.")

    context = context.unsqueeze(dim=0)

    checkpoint = save_hidden_states(model)

    full_tokens_bitmap_sorted = torch.zeros(1, device=DEVICE)
    pending_candidates_sorted = {'tokens': torch.empty(context.size(0), 0, device=DEVICE).long(),
                                 'scores': torch.zeros(1, device=DEVICE).float()}
    ready_candidates = update_ready_candidates(
        best_subtokens=torch.empty((0, 0), dtype=torch.long, device=DEVICE),
        best_scores=torch.empty((0), dtype=torch.float, device=DEVICE)
    )
    with torch.no_grad():
        while not topk_are_full_tokens(full_tokens_bitmap_sorted, top_k):
            new_candidates = get_new_candidates(model, context, beam_size, pending_candidates_sorted)

            all_candidates = {k: torch.cat([ready_candidates[k], new_candidates[k]], dim=0) for k in
                              ready_candidates.keys()}

            all_candidates['tokens'] = torch.cat([all_candidates['tokens'], all_candidates['num_tokens'][:, None]],
                                                 dim=-1)

            idx_in_score_asc_order = all_candidates['scores'].argsort()[:beam_size]
            best_candidates_sorted = {k: all_candidates[k][idx_in_score_asc_order] for k in all_candidates.keys()}

            full_tokens_bitmap_sorted = best_candidates_sorted['num_tokens'] < first_nonterm_token
            ready_candidate_idxs = full_tokens_bitmap_sorted.nonzero().squeeze(dim=1)
            pending_candidate_idxs = (full_tokens_bitmap_sorted == 0).nonzero().squeeze(dim=1)

            ready_candidates = update_ready_candidates(
                best_candidates_sorted['tokens'][ready_candidate_idxs],
                best_candidates_sorted['scores'][ready_candidate_idxs]
            )

            pending_candidates_sorted = {k: best_candidates_sorted[k][pending_candidate_idxs] for k in
                                         best_candidates_sorted.keys()}

            model[0].select_hidden(pending_candidates_sorted['hidden_indices'])
            context = pending_candidates_sorted['num_tokens'].unsqueeze(dim=1)

    model[0].hidden = checkpoint
    model[0].bs = checkpoint[0][0].size(0)

    return ready_candidates['tokens'][:top_k], ready_candidates['scores'][:top_k]
