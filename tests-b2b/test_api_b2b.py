import os
from pathlib import Path

import torch
import numpy as np

from codeprep.tokentypes.word import KeyWord
from langmodels import project_dir
from langmodels.evaluation.api import evaluate_on_path, evaluate_on_string
from langmodels.evaluation.result import EvaluationScenarioGrid, ShortEvaluation, EvaluationScenario, \
    ShortEvaluationResult
from langmodels.lmconfig.datamodel import Corpus, LMTrainingConfig, DeviceOptions, Training, RafaelsTrainingSchedule
from langmodels.model.context import ContextModifier
from langmodels.model.nn import take_hidden_state_snapshot
from langmodels.model.tokencategories import TokenCategory
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


def test_evaluate_model_on_path():
    actual = evaluate_on_path(load_default_model(),
                              Path(project_dir) /'data' /'dev' /'valid',
                              evaluation_scenario_grid=EvaluationScenarioGrid(context_modifier=ContextModifier(max_context_length=50), token_categories={TokenCategory.full_set()}),
                              batch_size=3)
    assert actual.scenarios[EvaluationScenario('perplexity')] \
           == ShortEvaluationResult(sum=26293.088098963723, n_samples=1522, exp=True)


def test_evaluate_model_on_file():
    actual = evaluate_on_path(load_default_model(),
                              Path(project_dir) /'data' /'dev' /'valid' /'StandardDataTypeEmitter.java',
                              evaluation_scenario_grid=EvaluationScenarioGrid(context_modifier=ContextModifier(max_context_length=50), token_categories={TokenCategory.full_set()}),
                              batch_size=3)

    assert len(actual.scenarios) == 1
    assert actual.scenarios[EvaluationScenario('perplexity')] == ShortEvaluationResult(sum=26293.088098963723, n_samples=1522, exp=True)


def test_evaluate_model_on_string():
    actual = evaluate_on_string(load_default_model(),
                                    'import java.lang,collections;',
                                evaluation_scenario_grid=EvaluationScenarioGrid(
                                        token_categories={TokenCategory.full_set(), TokenCategory.Builder().add({KeyWord}).build()}),
                                )

    assert len(actual.scenarios) == 2
    assert actual.scenarios[EvaluationScenario('perplexity', TokenCategory.Builder().add({KeyWord}).build())] \
           == ShortEvaluationResult(sum=8.018324851989746, n_samples=1, exp=True)

    assert actual.scenarios[EvaluationScenario('perplexity', TokenCategory.full_set())] \
           == ShortEvaluationResult(sum=57.25079941749573, n_samples=7, exp=True)


def test_train():
    np.random.seed(13)
    torch.manual_seed(13)

    model = train(comet=False, device_options=DeviceOptions(fallback_to_cpu=True),
          training_config=LMTrainingConfig(Corpus(os.path.join(project_dir, 'data', 'dev')),
                                           training=Training(schedule=RafaelsTrainingSchedule(max_epochs=1))))
    assert model.metrics.bin_entropy == 9.139766693115234
