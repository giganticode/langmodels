import unittest
from unittest.mock import patch

from langmodels.inference.entropies import get_entropy_for_each_line
from langmodels.model import TrainedModel
from tests.util import file_mock_with_lines


class GetEntropiesForEachFileIntegrationTest(unittest.TestCase):
    @patch('langmodels.inference.entropies.open')
    def test_simple(self, open_mock):
        # given
        test_trained_model = TrainedModel.get_default_model()
        mocked_file = file_mock_with_lines(['//button', 'class Button {'])
        open_mock.side_effect = [mocked_file]

        # when
        actual = get_entropy_for_each_line(test_trained_model, "a.java")
        self.assertEqual(2, len(actual))
        self.assertTrue(isinstance(actual[0], float))
        print(actual)


if __name__ == '__main__':
    unittest.main()
