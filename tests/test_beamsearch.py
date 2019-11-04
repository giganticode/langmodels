import os

import torch

from langmodels import project_dir
from langmodels.modelregistry import load_from_path


def test_hidden_state_not_changed_after_beam_search():
    trained_model = load_from_path(os.path.join(project_dir, 'data/models/dev_10k_1_10_190923.132328'))
    trained_model.feed_text("Hi there, it's")
    before = trained_model.model[0].hidden
    trained_model.predict_next_full_token(n_suggestions=10)
    after = trained_model.model[0].hidden

    for aa,bb in zip(after, before):
        for a, b in zip(aa, bb):
            assert torch.equal(b, a)