import unittest

import torch

from langmodels.modelregistry import load_model_by_name

DEFAULT_TEST_MODEL = 'langmodel-small-split_10k_1_512_190906.154943'


class TestBeamSearch(unittest.TestCase):
    def test_hidden_state_not_changed_after_beam_search(self):
        trained_model = load_model_by_name(DEFAULT_TEST_MODEL)
        trained_model.feed_text("Hi there, it's")
        before = trained_model.model[0].hidden
        trained_model.predict_next_full_token(n_suggestions=10)
        after = trained_model.model[0].hidden

        for aa,bb in zip(after, before):
            for a, b in zip(aa, bb):
                self.assertTrue(torch.equal(b, a))