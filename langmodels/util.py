import os
from math import log

from typing import Union, List

from langmodels.torchtypes import AnyDeviceFloatTensor


def to_binary_entropy(entropy: Union[float, AnyDeviceFloatTensor]) -> Union[float, AnyDeviceFloatTensor]:
    return entropy / log(2)


def distribute_into_chunks(x: int, max_chunk_value: int) -> List[int]:
    """
    >>> distribute_into_chunks(10, 4)
    [4, 3, 3]
    >>> distribute_into_chunks(7, 0)
    Traceback (most recent call last):
    ...
    ZeroDivisionError: integer division or modulo by zero
    >>> distribute_into_chunks(4, 9)
    [4]
    >>> distribute_into_chunks(12, 11)
    [6, 6]
    >>> distribute_into_chunks(0, 3)
    []
    >>> distribute_into_chunks(13, 3)
    [3, 3, 3, 2, 2]
    """
    if x == 0:
        return []
    n_chunks = (x + max_chunk_value - 1) // max_chunk_value
    lesser = x // n_chunks
    n_chunks_with_greater_value = x - n_chunks * lesser
    return [lesser + 1] * n_chunks_with_greater_value + [lesser] * (n_chunks - n_chunks_with_greater_value)


def split_list_into_nested_chunks(lst: List,
                                  elements_in_first_chunk: int,
                                  elements_in_chunk: int,
                                  max_elements_in_sub_chunk: int) -> List[List[List]]:
    """
    >>> split_list_into_nested_chunks([], 678, 678, 768)
    [[]]
    >>> split_list_into_nested_chunks([1], 678, 896, 567)
    [[[1]]]
    >>> split_list_into_nested_chunks([1], 0, 896, 567)
    Traceback (most recent call last):
    ...
    AssertionError
    >>> split_list_into_nested_chunks([1], 0, 0, 567)
    Traceback (most recent call last):
    ...
    ValueError: There can not be zero elements in a chunk or subchunk
    >>> split_list_into_nested_chunks([1], 0, 344, 0)
    Traceback (most recent call last):
    ...
    ValueError: There can not be zero elements in a chunk or subchunk
    >>> split_list_into_nested_chunks([1, 2, 3], 1, 1, 1)
    [[[1]], [[2]], [[3]]]
    >>> split_list_into_nested_chunks([1, 2, 3, 4, 5, 6, 7], 3, 3, 2)
    [[[1, 2], [3]], [[4, 5], [6]], [[7]]]
    >>> split_list_into_nested_chunks([1, 2, 3, 4, 5, 6, 7], 3, 3000000, 2)
    [[[1, 2], [3]], [[4, 5], [6, 7]]]
    """
    if elements_in_chunk == 0 or max_elements_in_sub_chunk == 0:
        raise ValueError("There can not be zero elements in a chunk or subchunk")

    elements_left = len(lst)
    chunk_sizes = [distribute_into_chunks(min(elements_left, elements_in_first_chunk), max_elements_in_sub_chunk)]
    elements_left -= elements_in_first_chunk
    next_chunk_length = min(elements_left, elements_in_chunk)
    while next_chunk_length > 0:
        chunk_sizes.append(distribute_into_chunks(next_chunk_length, max_elements_in_sub_chunk))
        elements_left -= next_chunk_length
        next_chunk_length = min(elements_left, elements_in_chunk)

    chunks = []
    current_start = 0
    for subchunk_sizes in chunk_sizes:
        subchunks = []
        for i in subchunk_sizes:
            subchunks.append(lst[current_start:current_start+i])
            current_start += i
        chunks.append(subchunks)

    if not (len(chunks[0]) > 0 or len(chunks) == 1):
        raise AssertionError()

    return chunks




HOME = os.environ['HOME']