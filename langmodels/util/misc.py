import os
from functools import reduce

from math import log

from torch import FloatTensor
from typing import Union, List, TypeVar, Sequence, Dict, Callable, Any, Iterable, Type, Set


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


K = TypeVar('K')
V = TypeVar('V')


def merge_dicts_(dict1: Dict[K, V], dict2: Dict[K, V], value_merger: Callable[[V, V], V]) -> Dict[K, V]:
    """
    this method returns modified `dict1`! and new words are added to the dictionary

    >>> dict1 = {"a": 3, "b": 4}
    >>> dict2 = {"b": 5, "c": 6}
    >>> merge_dicts_(dict1, dict2, value_merger=lambda x,y: x+y)
    {'a': 3, 'b': 9, 'c': 6}

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = v
        else:
            dict1[k] = value_merger(dict1[k], v)
    return dict1


def split_list_into_consequtive_chunks(lst: List[Any], n_chunks) -> List[List[Any]]:
    """
    >>> split_list_into_consequtive_chunks([1, 2, 3, 4, 5, 6, 7, 8], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8]]
    >>> split_list_into_consequtive_chunks([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5], [6, 7]]
    >>> split_list_into_consequtive_chunks([1, 2, 3, 4, 5, 6], 3)
    [[1, 2], [3, 4], [5, 6]]
    >>> split_list_into_consequtive_chunks([], 2)
    [[], []]
    >>> split_list_into_consequtive_chunks([1], 2)
    [[1], []]
    """
    result = []
    min_files_in_chunk = len(lst) // n_chunks
    chunks_with_aditional_file = len(lst) - min_files_in_chunk * n_chunks
    for i in range(chunks_with_aditional_file):
        result.append(lst[i * (min_files_in_chunk + 1):i * (min_files_in_chunk + 1) + (min_files_in_chunk + 1)])
    n_packed = (min_files_in_chunk+1) * chunks_with_aditional_file
    for i in range(n_chunks - chunks_with_aditional_file):
        result.append(lst[n_packed + i * min_files_in_chunk: n_packed + i * min_files_in_chunk + min_files_in_chunk])
    return result


def entropy_to_probability(entropy: float) -> float:
    """
    >>> entropy_to_probability(0.0)
    1.0

    >>> entropy_to_probability(1.0)
    0.5

    >>> entropy_to_probability(3.0)
    0.125

    >>> entropy_to_probability(100.0)
    7.888609052210118e-31
    """
    return 2 ** -entropy


def all_subclasses(classes: Iterable[Type]) -> Set[Type]:
    """
    >>> from codeprep.tokentypes.rootclasses import ParsedToken
    >>> subclasses = all_subclasses([ParsedToken])
    >>> sorted(map(lambda t: t.__name__, subclasses))
    ['ClosingBracket', 'ClosingCurlyBracket', 'Comment', 'Identifier', 'KeyWord', 'MultilineComment', 'NewLine', 'NonCodeChar', \
'NonEng', 'NonProcessibleToken', 'Number', 'One', 'OneLineComment', 'OpeningBracket', 'OpeningCurlyBracket', \
'Operator', 'ParsedToken', 'ProcessableTokenContainer', 'Semicolon', 'SpaceInString', 'SpecialToken', \
'StringLiteral', 'Tab', 'TextContainer', 'Whitespace', 'Zero']
    """
    return reduce(set.union, [{cls}.union(
        [s for c in cls.__subclasses__() for s in all_subclasses([c])])
        for cls in classes], set())


HOME = os.environ['HOME']

