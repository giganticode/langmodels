import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from langmodels import project_dir
from langmodels.evaluation.api import evaluate_on_path, evaluate_on_string, evaluate_on_file
from langmodels.lmconfig.datamodel import Corpus, LMTrainingConfig, DeviceOptions, Training, RafaelsTrainingSchedule
from langmodels.model.nn import take_hidden_state_snapshot
from langmodels.repository.load import load_default_model
from langmodels.training.training import train


def test_hidden_state_not_changed_after_evaluation():
    trained_model = load_default_model()
    trained_model.feed_text('Some context which after evaluation has to be restored', extension='java')
    before = take_hidden_state_snapshot(trained_model.model)
    evaluate_on_string(trained_model, 'import')
    after = take_hidden_state_snapshot(trained_model.model)

    for aa,bb in zip(after, before):
        for a, b in zip(aa, bb):
            assert torch.equal(b, a)


def test_evaluate_model_on_path_subtokens():
    f = tempfile.TemporaryDirectory()
    actual = evaluate_on_path(load_default_model(),
                              Path(project_dir) /'data' /'dev' /'valid',
                              save_to=Path(f.name), full_tokens=False,
                              batch_size=3)

    total = actual.total()
    assert int(total['Entropy']) == 9
    assert total['n_samples'] == 2839


def test_evaluate_model_on_path():
    f = tempfile.TemporaryDirectory()
    actual = evaluate_on_path(load_default_model(),
                              Path(project_dir) /'data' /'dev' /'valid',
                              save_to=Path(f.name),
                              batch_size=3)

    total = actual.total()
    assert int(total['Entropy']) == 16
    assert total['n_samples'] == 1647


def test_evaluate_model_on_file():
    actual = evaluate_on_file(load_default_model(),
                              Path(project_dir) /'data' /'dev' /'valid' /'StandardDataTypeEmitter.java')

    total = actual.total()
    assert int(total['Entropy']) == 15
    assert total['n_samples'] == 1528


def test_evaluate_model_on_string():
    actual = evaluate_on_string(load_default_model(),
                                'import java.lang.collections;')

    total = actual.total()
    assert int(total['Entropy']) == 8
    assert total['n_samples'] == 7


def test_train():
    np.random.seed(13)
    torch.manual_seed(13)

    model = train(comet=False, device_options=DeviceOptions(fallback_to_cpu=True),
          training_config=LMTrainingConfig(Corpus(os.path.join(project_dir, 'data', 'dev')),
                                           training=Training(schedule=RafaelsTrainingSchedule(max_epochs=1))))
    assert int(model.metrics.bin_entropy) == 9
