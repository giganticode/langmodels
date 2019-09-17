import unittest

import torch

from langmodels.model import TrainedModel


class TestBeamSearch(unittest.TestCase):
    def test_hidden_state_not_changed_after_beam_search(self):
        trained_model = TrainedModel.get_default_model()
        trained_model.feed_text("Hi there, it's")
        before = trained_model.model[0].hidden
        trained_model.predict_next_full_token(n_suggestions=10)
        after = trained_model.model[0].hidden

        for aa,bb in zip(after, before):
            for a, b in zip(aa, bb):
                self.assertTrue(torch.equal(b, a))