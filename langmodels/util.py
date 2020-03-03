import os
from math import log

from torch import FloatTensor
from typing import Union, List, TypeVar, Sequence


def to_binary_entropy(entropy: Union[float, FloatTensor]) -> Union[float, FloatTensor]:
    return entropy / log(2)


T = TypeVar('T')


def chunk_prepped_tokens(input_sequence: Sequence[T], first_chunk_max_length: int, max_context: int) -> List[Sequence[T]]:
    """
    >>> chunk_prepped_tokens([], 8, 200)
    [[]]
    >>> chunk_prepped_tokens([1, 2], 8, 200)
    [[1, 2]]
    >>> chunk_prepped_tokens([1, 2], 1, 200)
    [[1], [2]]
    """
    elements_to_pack_in_first_chunk = min(first_chunk_max_length, len(input_sequence))
    chunked_prepped_tokens = [input_sequence[:elements_to_pack_in_first_chunk]]

    packed_elements = elements_to_pack_in_first_chunk
    while packed_elements < len(input_sequence):
        elements_to_pack = min(max_context, len(input_sequence) - packed_elements)
        chunked_prepped_tokens.append(input_sequence[packed_elements:packed_elements + elements_to_pack])
        packed_elements += elements_to_pack

    return chunked_prepped_tokens


HOME = os.environ['HOME']