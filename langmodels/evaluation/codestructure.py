import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

from codeprep.util.misc import cum_sum


@dataclass
class CodeSnippetStructure:
    """
    >>> snippet_a = CodeSnippetStructure(Path(''), [3], 2)
    >>> snippet_a.split(4)
    Traceback (most recent call last):
    ...
    IndexError: Token 4 is out of bounds of this snippet
    >>> snippet_a.split(0)
    (.: [], first-line: 2, .: [3], first-line: 2)
    >>> snippet_a.split(2)
    (.: [2], first-line: 2, .: [1], first-line: 2)
    >>> snippet_a.split(3)
    Traceback (most recent call last):
    ...
    IndexError: Token 3 is out of bounds of this snippet

    >>> snippet_b = CodeSnippetStructure(Path(''), [3, 4], 2)
    >>> snippet_b.split(3)
    (.: [3], first-line: 2, .: [4], first-line: 3)
    >>> snippet_b.split(4)
    (.: [3, 1], first-line: 2, .: [3], first-line: 3)
    >>> snippet_b.split(7)
    Traceback (most recent call last):
    ...
    IndexError: Token 7 is out of bounds of this snippet

    """
    path: Path
    subtokens_in_each_line: List[int]
    first_line: int

    def merge(self, other: 'CodeSnippetStructure') -> 'CodeSnippetStructure':
        if self.path != other.path:
            raise ValueError("Cannot merge two different files.")

        last_line_of_first_snippet = self.first_line + len(self.subtokens_in_each_line) - 1
        if last_line_of_first_snippet + 1 == other.first_line:
            lines_combines = self.subtokens_in_each_line + other.subtokens_in_each_line
        elif last_line_of_first_snippet == other.first_line:
            lines_combines = self.subtokens_in_each_line[:-1] + \
                             [self.subtokens_in_each_line[-1] + other.subtokens_in_each_line[0]] + \
                             other.subtokens_in_each_line[1:]
        else:
            raise ValueError("Prepped files are not adjecent.")

        return CodeSnippetStructure(self.path, lines_combines, self.first_line)

    def split(self, second_part_start_index: int) -> Tuple['CodeSnippetStructure', 'CodeSnippetStructure']:
        cumulative_lengths = cum_sum(self.subtokens_in_each_line)
        line_to_be_split = bisect.bisect_right(cumulative_lengths, second_part_start_index, 0, len(cumulative_lengths))
        total_lengths_of_previous_lines = cumulative_lengths[line_to_be_split-1] if line_to_be_split > 0 else 0
        position_to_split_in_line = second_part_start_index - total_lengths_of_previous_lines
        if line_to_be_split < len(cumulative_lengths):
            lines_in_first = self.subtokens_in_each_line[:line_to_be_split]
            if position_to_split_in_line > 0:
                lines_in_first.append(position_to_split_in_line)
            lines_in_second = [self.subtokens_in_each_line[line_to_be_split] - position_to_split_in_line] + self.subtokens_in_each_line[line_to_be_split+1:]
            first = CodeSnippetStructure(self.path, lines_in_first, self.first_line)
            second = CodeSnippetStructure(self.path, lines_in_second, self.first_line + line_to_be_split)
            return first, second
        else:
            raise IndexError(f"Token {second_part_start_index} is out of bounds of this snippet")

    def __len__(self) -> int:
        return sum(self.subtokens_in_each_line)

    def __repr__(self):
        return f'{self.path}: {self.subtokens_in_each_line}, first-line: {self.first_line}'


@dataclass
class CodeBaseStructure:
    """
    >>> snippet = CodeSnippetStructure(Path(''), [3, 4], 2)
    >>> snippet_a, snippet_b = snippet.split(5)
    >>> prepped_code = CodeBaseStructure([snippet_a, snippet_b])
    >>> prepped_code.split(2)
    (CodeBaseStructure(snippets=[.: [2], first-line: 2]), CodeBaseStructure(snippets=[.: [1, 2], first-line: 2, .: [2], first-line: 3]))

    """
    snippets: List[CodeSnippetStructure] = field(default_factory=list)

    def add_snippet(self, prepped_snippet: CodeSnippetStructure) -> 'CodeBaseStructure':
        if not self.snippets or self.snippets[-1].path != prepped_snippet.path:
            self.snippets.append(prepped_snippet)
        else:
            self.snippets[-1] = self.snippets[-1].merge(prepped_snippet)
        return self

    def merge(self, code_base_structure: 'CodeBaseStructure') -> 'CodeBaseStructure':
        for snippet in code_base_structure.snippets:
            self.add_snippet(snippet)
        return self

    def _get_cumularive_snippet_lengths(self) -> List[int]:
        return cum_sum(map(lambda x: len(x), self.snippets))

    def split(self, second_part_start_index: int) -> Tuple['CodeBaseStructure', 'CodeBaseStructure']:
        cumulative_lengths = self._get_cumularive_snippet_lengths()
        snippet_to_be_split = bisect.bisect_right(cumulative_lengths, second_part_start_index, 0, len(cumulative_lengths))
        total_lengths_of_previous_snippets = cumulative_lengths[snippet_to_be_split-1] if snippet_to_be_split > 0 else 0
        position_to_split_in_snippet = second_part_start_index - total_lengths_of_previous_snippets
        if snippet_to_be_split < len(cumulative_lengths):
            first, second = self.snippets[snippet_to_be_split].split(position_to_split_in_snippet)
            snippets_in_first = self.snippets[:snippet_to_be_split]
            if len(first) > 0:
                snippets_in_first.append(first)
            snippets_in_second = [second] + self.snippets[snippet_to_be_split+1:]
            return CodeBaseStructure(snippets_in_first), CodeBaseStructure(snippets_in_second)
        else:
            raise IndexError(f"Token {second_part_start_index} is out of bounds")

    def pop(self) -> Optional[CodeSnippetStructure]:
        if len(self.snippets)> 1:
            return self.snippets.pop(0)
        else:
            return None

    def pop_last(self) -> CodeSnippetStructure:
        assert len(self.snippets) == 1

        return self.snippets.pop()

    def __len__(self) -> int:
        return sum(map(lambda x: len(x), self.snippets))


@dataclass(frozen=True)
class CodeLocation:
    path: Path
    line: int