from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Mapping, Generator, Union

import torch
from torch.nn.functional import cross_entropy

from codeprep.preprocess.tokens import TokenSequence
from codeprep.preprocess.codestructure import CodeBaseStructure, SnippetStructure
from langmodels.evaluation.dataloader import BatchedTokenLoader
from langmodels.model.model import TrainedModel
from langmodels.model.nn import get_last_layer_activations
from langmodels.util.cuda import get_device

LossesWithMetadata = Tuple[List[List[float]], List[CodeBaseStructure], List[TokenSequence]]


@dataclass
class LossesWithMetadata:
    losses: torch.Tensor
    code_structure: CodeBaseStructure
    prepped_tokens: TokenSequence

    def __post_init__(self):
        if not len(self.losses) == self.prepped_tokens.sub_token_size():
            raise AssertionError('')

    @classmethod
    def empty(cls, device: Union[int, str]) -> 'LossesWithMetadata':
        return cls(torch.tensor([], device=device), CodeBaseStructure.empty(), TokenSequence.empty())

    def extend(self, other: 'LossesWithMetadata') -> None:
        """
        >>> LossesWithMetadata(torch.tensor([1.]), CodeBaseStructure.empty(), TokenSequence.empty())
        Traceback (most recent call last):
        ...
        AssertionError
        >>> l = LossesWithMetadata.empty('cpu')
        >>> class TypeA: pass
        >>> from codeprep.preprocess.metadata import PreppedTokenMetadata
        >>> token_seq = TokenSequence.create(['hi</t>', 'the' ,'re</t>'], PreppedTokenMetadata([1, 2], [TypeA, TypeA]), full_token_view=True)
        >>> code_base_structure = CodeBaseStructure.of([SnippetStructure.from_path_and_lines(Path(''), [2], 0)])
        >>> l.extend(LossesWithMetadata(torch.tensor([0.1, 0.2], device='cpu'), code_base_structure, token_seq.sub_token_view()[:2]))
        >>> l
        LossesWithMetadata(losses=tensor([0.1000, 0.2000]), code_structure=CodeBaseStructure(snippets=[.: [2], first-line: 0]), prepped_tokens=[['hi</t>'], ['the']])
        >>> l.split_by_first_subtoken()
        (LossesWithMetadata(losses=tensor([0.1000]), code_structure=CodeBaseStructure(snippets=[.: [1], first-line: 0]), prepped_tokens=[['hi</t>']]), LossesWithMetadata(losses=tensor([0.2000]), code_structure=CodeBaseStructure(snippets=[.: [1], first-line: 0]), prepped_tokens=[['the']]))
        >>> l.extend(LossesWithMetadata(torch.tensor([0.4]), CodeBaseStructure.of([SnippetStructure.from_path_and_lines(Path(''), [1], 0)]), token_seq.sub_token_view()[2:]))
        >>> l.split_by_first_subtoken()
        (LossesWithMetadata(losses=tensor([0.1000, 0.2000, 0.4000]), code_structure=CodeBaseStructure(snippets=[.: [3], first-line: 0]), prepped_tokens=[['hi</t>'], ['the', 're</t>']]), LossesWithMetadata(losses=tensor([]), code_structure=CodeBaseStructure(snippets=[]), prepped_tokens=[]))
        """
        self.losses = torch.cat([self.losses, other.losses], dim=-1)
        self.code_structure.merge(other.code_structure)
        self.prepped_tokens.extend(other.prepped_tokens)

    def split_by_first_subtoken(self) -> Tuple['LossesWithMetadata', 'LossesWithMetadata']:
        split_on_full_token = self.prepped_tokens.full_token_size()
        if not self.prepped_tokens.is_complete():
            split_on_full_token -= 1
        full_part = self.prepped_tokens.full_token_view()[:split_on_full_token]
        other_part = self.prepped_tokens.full_token_view()[split_on_full_token:]
        split_on = full_part.sub_token_size()
        f, s = self.code_structure.split(split_on)
        return LossesWithMetadata(self.losses[:split_on], f, full_part), \
               LossesWithMetadata(self.losses[split_on:], s, other_part)


def cut_off_placeholders(losses_with_metadata: List[LossesWithMetadata], non_max_seq_len: Mapping[int, int]) -> List[LossesWithMetadata]:
    cut_off_losses = losses_with_metadata
    for i, actual_seq_len in non_max_seq_len.items():
        cut_off_losses[i] = LossesWithMetadata(losses_with_metadata[i].losses[0:actual_seq_len],
                                                 losses_with_metadata[i].code_structure,
                                                 losses_with_metadata[i].prepped_tokens.sub_token_view()[:actual_seq_len])
    return cut_off_losses


def calculate_losses_for_batch(trained_model: TrainedModel, token_loader: BatchedTokenLoader) -> Generator[List[LossesWithMetadata], None, None]:
    """
    changes hidden states of the model!!
    """
    with torch.no_grad():
        batch_size = token_loader.batch_size
        numericalized_start_point = trained_model.vocab.stoi[trained_model.STARTING_TOKEN]
        numericalized_last_predicted = torch.full((batch_size, 1), numericalized_start_point, dtype=torch.int64, device=get_device())

        losses_with_metadata_list = [LossesWithMetadata.empty(get_device()) for i in range(batch_size)]

        for prepped_token_batch, non_max_seq_len, code_structure, reset in token_loader:
            sub_token_seq_len = prepped_token_batch[0].sub_token_size()

            numericalized_batch = torch.tensor([trained_model.vocab.numericalize(sequence)
                                                for i, sequence in enumerate(prepped_token_batch)],
                                               device=get_device(), dtype=torch.int64)

            input_batch = torch.cat([numericalized_last_predicted, numericalized_batch[:, :-1]], dim=1)
            last_layer = get_last_layer_activations(trained_model.model, input_batch)
            loss: torch.Tensor = cross_entropy(last_layer.view(-1, last_layer.shape[-1]),
                                               numericalized_batch.view(-1),
                                               reduction='none').view(-1, sub_token_seq_len)
            numericalized_last_predicted = numericalized_batch[:, -1:]

            current_batch_losses_with_metadata = [LossesWithMetadata(loss[i], code_structure[i], prepped_token_batch[i]) for i in range(batch_size)]
            current_batch_losses_with_metadata = cut_off_placeholders(current_batch_losses_with_metadata, non_max_seq_len)

            for i in range(batch_size):
                losses_with_metadata_list[i].extend(current_batch_losses_with_metadata[i])

            to_yield = []
            for i in range(batch_size):
                y, losses_with_metadata_list[i] = losses_with_metadata_list[i].split_by_first_subtoken()
                to_yield.append(y)

            yield to_yield

            if reset:
                trained_model.reset()