from typing import List
from unittest.mock import MagicMock


def file_mock_with_lines(lines: List[str]):
    file_mock = MagicMock(spec=['__enter__', '__exit__'])
    handle1 = file_mock.__enter__.return_value
    handle1.__iter__.return_value = iter(map(lambda l: l + '\n', lines))
    return file_mock