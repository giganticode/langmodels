import unittest
import torch

from langmodels.metrics import mrr

class MrrTest(unittest.TestCase):
    def test_mrr_1(self):
        preds = torch.tensor([[[0.2, 0.55, 0.25], [0.005, 0.005, 0.99]]])
        targs = torch.tensor([[1, 2]])

        actual = mrr(preds, targs)
        expected = 1.0
        self.assertAlmostEqual(expected, actual.item())

    def test_mrr_simple(self):
        preds = torch.tensor([[[0.2, 0.55, 0.25], [0.006, 0.004, 0.99]]])
        targs = torch.tensor([[0, 0]])

        actual = mrr(preds, targs)
        expected = 0.41666668653
        self.assertAlmostEqual(expected, actual.item())


if __name__ == '__main__':
    unittest.main()