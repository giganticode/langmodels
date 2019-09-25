import unittest

from dataprep.parse.model.placeholders import placeholders
from fastai.text import Vocab

from langmodels.model import _create_term_vocab

cpe = placeholders['compound_word_end']


class CreateTermVocabTest(unittest.TestCase):
    def test_simple_vocab(self):
        input_vocab = Vocab(['a', f'b{cpe}', 'c', 'd', f'e{cpe}'])
        actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

        expected_itos = [f'b{cpe}', f'e{cpe}', 'a', 'c', 'd']

        self.assertEqual(expected_itos, actual_vocab.itos)
        self.assertEqual(2, actual_first_non_term_index)

    def test_empty_vocab(self):
        input_vocab = Vocab([])
        actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

        expected_itos = []

        self.assertEqual(expected_itos, actual_vocab.itos)
        self.assertEqual(0, actual_first_non_term_index)

    def test_only_terminals(self):
        input_vocab = Vocab([f'b{cpe}', f'e{cpe}'])
        actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

        expected_itos = [f'b{cpe}', f'e{cpe}']

        self.assertEqual(expected_itos, actual_vocab.itos)
        self.assertEqual(2, actual_first_non_term_index)

    def test_only_non_terminals(self):
        input_vocab = Vocab(['a', 'c', 'd'])
        actual_vocab, actual_first_non_term_index = _create_term_vocab(input_vocab)

        expected_itos = ['a', 'c', 'd']

        self.assertEqual(expected_itos, actual_vocab.itos)
        self.assertEqual(0, actual_first_non_term_index)


if __name__ == '__main__':
    unittest.main()
