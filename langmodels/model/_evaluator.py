from typing import List, Tuple

import torch
from torch.nn.functional import cross_entropy

from codeprep.preprocess.tokens import TokenSequence
from codeprep.preprocess.codestructure import CodeBaseStructure
from langmodels.evaluation.dataloader import BatchedTokenLoader
from langmodels.model.model import TrainedModel
from langmodels.model.nn import get_last_layer_activations
from langmodels.util.cuda import get_device

LossesWithMetadata = Tuple[List[List[float]], List[CodeBaseStructure], List[TokenSequence]]


def reset_losses_with_metadata_lists(batch_size: int) -> LossesWithMetadata:
    all_losses = [[] for i in range(batch_size)]
    all_code_structure = [CodeBaseStructure([]) for i in range(batch_size)]
    all_prepped_tokens = [TokenSequence.empty() for i in range(batch_size)]

    return all_losses, all_code_structure, all_prepped_tokens


def calculate_losses(trained_model: TrainedModel, token_loader: BatchedTokenLoader) -> LossesWithMetadata:
    """
    changes hidden states of the model!!
    """
    batch_size = token_loader.batch_size
    numericalized_start_point = trained_model.vocab.stoi[trained_model.STARTING_TOKEN]
    numericalized_last_predicted = torch.full((batch_size, 1), numericalized_start_point, dtype=torch.int64, device=get_device())

    all_losses, all_code_structure, all_prepped_tokens = reset_losses_with_metadata_lists(batch_size)

    for prepped_token_batch, non_max_seq_len, code_structure, reset in token_loader:
        sub_token_seq_len = prepped_token_batch[0].sub_token_size()

        numericalized_batch = torch.tensor([trained_model.vocab.numericalize(sequence)
                                            for i, sequence in enumerate(prepped_token_batch)],
                                           device=get_device(), dtype=torch.int64)

        input_batch = torch.cat([numericalized_last_predicted, numericalized_batch[:, :-1]], dim=1)
        last_layer = get_last_layer_activations(trained_model.model, input_batch)
        loss = cross_entropy(last_layer.view(-1, last_layer.shape[-1]),
                               numericalized_batch.view(-1),
                               reduction='none').view(-1, sub_token_seq_len)
        numericalized_last_predicted = numericalized_batch[:, -1:]

        for i in range(batch_size):
            actual_seq_len = non_max_seq_len[i] if i in non_max_seq_len else sub_token_seq_len
            all_losses[i].extend(loss[i, 0:actual_seq_len].tolist())
            all_code_structure[i].merge(code_structure[i])
            all_prepped_tokens[i] = all_prepped_tokens[i] + prepped_token_batch[i].sub_token_view()[:actual_seq_len]

        if reset:
            trained_model.reset()
            yield all_losses, all_code_structure, all_prepped_tokens

            all_losses, all_code_structure, all_prepped_tokens = reset_losses_with_metadata_lists(batch_size)