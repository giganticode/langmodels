import numpy as np
from fastai.text import Vocab, TextList
from pathlib import Path
from unittest.mock import patch

from langmodels.training.numericalize import Numericalizer
from tests.file_test_util import file_mock_with_lines


def all_trues(bools):
    try:
        fltn_bools = np.hstack(bools)
    except: # should not pass silently.
        fltn_bools = np.array(all_trues(a) for a in bools)
    ints = fltn_bools.prod()
    if isinstance(ints, np.ndarray):
        return all_trues(ints)
    return bool(ints)


@patch('langmodels.training.training.open')
def test_simple(open_mock):
    # Given
    file_mock1 = file_mock_with_lines(['1', 'My Class'])
    file_mock2 = file_mock_with_lines(['1', 'hi'])

    open_mock.side_effect = [file_mock1, file_mock2]

    numericalizer = Numericalizer(Vocab(['`unk', '`pad', '1', 'My', 'Class', 'hi']), n_cpus=1)

    text_list = TextList([Path('/path/to/some/file1'), Path('/path/to/some/file2')])

    # when
    numericalizer.process(text_list)

    # then
    expected = np.array([np.array([2, 3, 4]), np.array([2, 5])])
    assert all_trues(np.equal(expected, text_list.items, dtype=np.object))
