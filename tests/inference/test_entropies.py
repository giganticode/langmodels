import unittest

from langmodels.inference.entropies import word_entropy_list, subword_average, word_average


class WordEntropyListTest(unittest.TestCase):
    def test_invalid_empty_boundary_list(self):
        with self.assertRaises(ValueError):
            word_entropy_list([], [])

    def test_invalid_non_empty_boundary_list(self):
        with self.assertRaises(ValueError):
            word_entropy_list([0.1, 0.7, 8.6], [0, 2])

    def test_empty(self):
        self.assertEqual([], word_entropy_list([], [0]))

    def test_all_full_words(self):
        self.assertEqual([1.0, 2.0, 3.0], word_entropy_list([1.0, 2.0, 3.0], [0, 1, 2, 3]))

    def test_some_full_words(self):
        self.assertEqual([1.0, 5.0], word_entropy_list([1.0, 2.0, 3.0], [0, 1, 3]))

    def test_one_full_word(self):
        self.assertEqual([6.0], word_entropy_list([1.0, 2.0, 3.0], [0, 3]))


class SubwordAverageTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(.0, subword_average([], [0]))

    def test_not_empty(self):
        self.assertEqual(2.0, subword_average([1.0, 2.0, 3.0], [0, 1, 3]))


class WordAverageTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(.0, word_average([], [0]))

    def test_not_empty(self):
        self.assertEqual(3.0, word_average([1.0, 2.0, 3.0], [0, 1, 3]))


if __name__ == '__main__':
    unittest.main()