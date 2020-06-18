import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Any, Optional, Tuple, Mapping, Iterator

import sys

from codeprep.api.text import basic
from codeprep.preprocess.codestructure import CodeBaseStructure
from codeprep.preprocess.metadata import PreppedTokenMetadata
from codeprep.preprocess.tokens import TokenSequence
from codeprep.tokentypes.word import SpecialToken
from langmodels.model.context import ContextModifier
from langmodels.util.file import read_file_contents, get_all_files
from langmodels.util.misc import split_list_into_consequtive_chunks

DEFAULT_MAX_SIZE = 500


def split_without_looking_at_files(all_files: List[Any], batch_size: int) -> List[List[Any]]:
    return split_list_into_consequtive_chunks(all_files, batch_size)


def split_with_looking_at_files(all_files: List[Any], batch_size: int, func: Callable[[Path], int]) -> List[List[Any]]:
    """
    >>> split_with_looking_at_files([1, 2, 6], 3, lambda x: x)
    [[1, 2], [6], []]
    >>> split_with_looking_at_files([1], 3, lambda x: x)
    [[1], [], []]
    >>> split_with_looking_at_files([4, 2, 5, 1, 6], 3, lambda x: x)
    [[4, 2], [5, 1], [6]]
    """
    cumul_sizes = []
    sum_ = 0
    for file in all_files:
        sum_ += func(file)
        cumul_sizes.append(sum_)
    size_per_chunk = cumul_sizes[-1] // batch_size + 1
    chunked_files = []
    last_start = 0
    for i in range(1, batch_size):
        pos = bisect_right(cumul_sizes, size_per_chunk * i)
        if pos <= last_start:
            pos = last_start + 1
        chunked_files.append(all_files[last_start:pos])
        last_start = pos
    chunked_files.append(all_files[last_start:])
    assert len(all_files) == sum(map(lambda x: len(x), chunked_files))
    return chunked_files


def split_into_batches(all_files: List[Path], batch_size: int) -> List[List[Path]]:
    if len(all_files) > batch_size * 100:
        return split_without_looking_at_files(all_files, batch_size)
    else:
        return split_with_looking_at_files(all_files, batch_size, func=os.path.getsize)


class BatchedFileLoader:
    def __init__(self, path: Path, batch_size: int, extensions_supported: List[str]):
        self.all_files = [f for f in get_all_files(str(path))]

        self.batch_size = batch_size if len(self.all_files) > batch_size else len(self.all_files)
        self.file_batches = split_into_batches(self.all_files, self.batch_size)

        self.max_batch = max(map(lambda b: len(b), self.file_batches))
        self.current = 0

        self.extensions_supported = extensions_supported

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_batch

    def __next__(self) -> List[Tuple[Path, str]]:
        if self.current < len(self):
            result = [(file_batch[self.current], read_file_contents(file_batch[self.current]))
                      if self.current < len(file_batch) else None for file_batch in self.file_batches]
            self.current += 1
            return result
        else:
            raise StopIteration

    def should_append_eof(self):
        return True


@dataclass
class AllTextLoader(BatchedFileLoader):
    text_list: Optional[List[str]]
    extension: str = 'java'
    append_eof: bool = False

    def __post_init__(self):
        self.all_files = [Path(f'{i}.java') for i in range(len(self.text_list))]

    def n_total_files(self) -> int:
        return len(self.text_list)

    def __iter__(self) -> Iterator[List[Tuple[Optional[Path], str]]]:
        return self

    def should_append_eof(self):
        return self.append_eof

    def __next__(self) -> List[Tuple[Optional[Path], str]]:
        if self.text_list:
            to_return = [(Path(f'{i}.java'), text) for i, text in enumerate(self.text_list)]
            self.text_list = None
            return to_return
        else:
            raise StopIteration

    def __len__(self):
        return 1

    @property
    def batch_size(self):
        return len(self.text_list)


class BatchedTokenLoader:
    PLACEHOLDER = '`pad'

    #TODO add context modifier

    def __init__(self, batch_file_loader: BatchedFileLoader, prep_function: Callable, max_seq_len: Optional[int] = None,
                 context_modifier: Optional[ContextModifier] = None, return_file_structure: bool = False):
        """
        >>> batch_file_loader = AllTextLoader(['class A', 'public getName {'], append_eof=True)
        >>> batched_token_loader = BatchedTokenLoader(batch_file_loader, basic, max_seq_len=2, context_modifier=ContextModifier(max_context_length=2))
        >>> iterator = iter(batched_token_loader)
        >>> next(iterator)
        ([['class', 'A'], ['public', '<w>']], {}, \
[CodeBaseStructure(snippets=[0.java: [2], first-line: 0]), CodeBaseStructure(snippets=[1.java: [2], first-line: 0])], False)
        >>> next(iterator)
        ([['`pad', '`pad'], ['get', 'Name']], {0: 0}, \
[CodeBaseStructure(snippets=[]), CodeBaseStructure(snippets=[1.java: [2], first-line: 0])], False)
        >>> next(iterator)
        ([['`pad'], ['</w>']], {0: 0}, \
[CodeBaseStructure(snippets=[]), CodeBaseStructure(snippets=[1.java: [1], first-line: 0])], True)
        >>> next(iterator)
        ([['<EOF>', '`pad'], ['{', '<EOF>']], {0: 1}, \
[CodeBaseStructure(snippets=[0.java: [1], first-line: 0]), CodeBaseStructure(snippets=[1.java: [2], first-line: 0])], True)
        >>> next(iterator)
        Traceback (most recent call last):
        ...
        StopIteration
        """
        self.n_file_batches = len(batch_file_loader)
        self.n_files = len(batch_file_loader.all_files)
        self.batch_file_loader_iter: Iterator[List[Tuple[Optional[Path], str]]] = iter(batch_file_loader)
        self.append_eof: bool = batch_file_loader.should_append_eof()
        self.prep_function: Callable[[Any], TokenSequence] = prep_function
        self.max_seq_len: int = max_seq_len or DEFAULT_MAX_SIZE
        self.batch_size: int = batch_file_loader.batch_size
        self.reset_every: int = context_modifier.max_context_length if context_modifier else sys.maxsize
        self.tokens_after_reset: List[int] = [0 for _ in range(self.batch_size)]

        self.buffer: List[TokenSequence] = [TokenSequence.empty() for _ in range(batch_file_loader.batch_size)]
        self.code_structures: List[CodeBaseStructure] = [CodeBaseStructure.empty() for _ in range(batch_file_loader.batch_size)]
        self.can_still_load_files: bool = True

        self.tokens_are_finished: bool = False

    def _need_to_load_files(self) -> bool:
        for i, chunk in enumerate(self.buffer):
            if chunk.sub_token_size() < self.max_seq_len and chunk.full_token_size() < self.reset_every - self.tokens_after_reset[i]:
                return True
        return False

    def __iter__(self) -> Iterator[Tuple[List[TokenSequence], Mapping[int, int], List[CodeBaseStructure], bool]]:
        return self

    def _load_file_batch(self) -> None:
        new_files = next(self.batch_file_loader_iter)
        for i, path_and_text in enumerate(new_files):
            if path_and_text is None:
                continue
            path, text = path_and_text
            extension = path.suffix[1:]
            prepped_text, code_snippet_structure = self.prep_function(text, extension, append_eof=self.append_eof, path=path)
            self.code_structures[i].add_snippet(code_snippet_structure)
            self.buffer[i] = self.buffer[i].extend(prepped_text) if len(self.buffer[i]) > 0 else prepped_text

    def _get_new_sequence(self, i: int) -> TokenSequence:
        n_full_tokens_needed = self.reset_every-self.tokens_after_reset[i]
        full_tokens_until_next_reset = self.buffer[i].full_token_view()[:n_full_tokens_needed]
        n_subtokens_until_reset = full_tokens_until_next_reset.sub_token_size()
        if n_subtokens_until_reset > self.max_seq_len:
            actual_seq_len = self.max_seq_len
        else:
            actual_seq_len = n_subtokens_until_reset
        new_batch = full_tokens_until_next_reset.sub_token_view()[:actual_seq_len]
        self.buffer[i] = self.buffer[i].sub_token_view()[actual_seq_len:]
        actual_seq_len_full_tokens = new_batch.full_token_size()
        if new_batch.ends_with_incomplete_token:
            actual_seq_len_full_tokens -= 1
        self.tokens_after_reset[i] += actual_seq_len_full_tokens
        return new_batch

    def __next__(self) -> Tuple[List[TokenSequence], Mapping[int, int], List[CodeBaseStructure], bool]:
        if self.tokens_are_finished:
            raise StopIteration

        while self.can_still_load_files and self._need_to_load_files():
            try:
                self._load_file_batch()
            except StopIteration:
                self.can_still_load_files = False
        min_non_empty_chunk_len = self._min_non_empty_buffer_len()
        if min_non_empty_chunk_len == 0:
            raise StopIteration

        result, code_structure = [], []
        reset_context, buffers_are_empty = True, True
        for i in range(self.batch_size):
            result.append(self._get_new_sequence(i))

            if self.tokens_after_reset[i] < self.reset_every and self.buffer[i].sub_token_size() > 0:
                reset_context = False
            if self.buffer[i].sub_token_size() > 0:
                buffers_are_empty = False

        for i in range(self.batch_size):
            code_structure_to_return, self.code_structures[i] = self.code_structures[i].split(result[i].sub_token_size())
            code_structure.append(code_structure_to_return)

        non_max_seq_lens = {}
        max_actual_seq_len = max(map(lambda r: r.sub_token_size(), result))
        for i in range(self.batch_size):
            n_padding_tokens = max_actual_seq_len - result[i].sub_token_size()
            if n_padding_tokens > 0:
                non_max_seq_lens[i] = result[i].sub_token_size()

                padding_tokens = [BatchedTokenLoader.PLACEHOLDER] * n_padding_tokens
                padding_token_metadata = PreppedTokenMetadata([1] * n_padding_tokens, [SpecialToken] * n_padding_tokens)
                padding_token_seq = TokenSequence.of(padding_tokens, padding_token_metadata)
                result[i] = result[i].extend(padding_token_seq)

        if reset_context:
            self.tokens_after_reset: List[int] = [0 for _ in range(self.batch_size)]

        if buffers_are_empty and not self.can_still_load_files:
            reset_context = True
            self.tokens_are_finished = True

        return result, non_max_seq_lens, code_structure, reset_context

    def _min_non_empty_buffer_len(self) -> int:
        non_empty_chunks = list(filter(lambda x: x.sub_token_size() > 0, self.buffer))
        if non_empty_chunks:
            return min(map(lambda x: x.sub_token_size(), non_empty_chunks))
        else:
            return 0

    @classmethod
    def from_path(cls, path: Path, prep_func: Callable, batch_size: int, return_file_structure: bool, context_modifier: ContextModifier = None) -> 'BatchedTokenLoader':
        return BatchedTokenLoader(BatchedFileLoader(path, batch_size, ['java']), prep_func,
                                  context_modifier=context_modifier or ContextModifier(),
                                  return_file_structure=return_file_structure)

    @classmethod
    def from_file(self, path: Path, prep_func: Callable, return_file_structure: bool) -> 'BatchedTokenLoader':
        return BatchedTokenLoader(BatchedFileLoader(path, 1, [path.suffix[1:]]), prep_func, return_file_structure=return_file_structure)

    @classmethod
    def from_text(cls, text: str, prep_func: Callable, extension: str, append_eof: bool) -> 'BatchedTokenLoader':
        return BatchedTokenLoader(AllTextLoader([text], extension, append_eof), prep_func, return_file_structure=False)
