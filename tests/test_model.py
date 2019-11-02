from dataprep.parse.model.placeholders import placeholders
from fastai.text import Vocab

from langmodels.model import _create_term_vocab

cpe = placeholders['compound_word_end']


def test_simple_vocab():
    input_vocab = Vocab(['a', f'b{cpe}', 'c', 'd', f'e{cpe}'])
    actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

    expected_itos = [f'b{cpe}', f'e{cpe}', 'a', 'c', 'd']

    assert expected_itos == actual_vocab.itos
    assert 2 == actual_first_non_term_index


def test_empty_vocab():
    input_vocab = Vocab([])
    actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

    expected_itos = []

    assert expected_itos == actual_vocab.itos
    assert 0 == actual_first_non_term_index


def test_only_terminals():
    input_vocab = Vocab([f'b{cpe}', f'e{cpe}'])
    actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

    expected_itos = [f'b{cpe}', f'e{cpe}']

    assert expected_itos == actual_vocab.itos
    assert 2 == actual_first_non_term_index


def test_only_non_terminals():
    input_vocab = Vocab(['a', 'c', 'd'])
    actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

    expected_itos = ['a', 'c', 'd']

    assert expected_itos == actual_vocab.itos
    assert 0 == actual_first_non_term_index
