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

    scenario = actual.scenarios[EvaluationScenario('perplexity')]
    assert int(scenario.sum) == 26258
    assert scenario.n_samples == 1528
    assert scenario.exp


def test_evaluate_model_on_file():
    actual = evaluate_on_path(load_default_model(),
                              Path(project_dir) /'data' /'dev' /'valid' /'StandardDataTypeEmitter.java',
                              evaluation_scenario_grid=EvaluationScenarioGrid(context_modifier=ContextModifier(max_context_length=50), token_categories={TokenCategory.full_set()}),
                              batch_size=3)

    assert len(actual.scenarios) == 1
    scenario = actual.scenarios[EvaluationScenario('perplexity')]
    assert int(scenario.sum) == 26258
    assert scenario.n_samples == 1528
    assert scenario.exp


def test_evaluate_model_on_string():
    actual = evaluate_on_string(load_default_model(),
                                    'import java.lang,collections;',
                                evaluation_scenario_grid=EvaluationScenarioGrid(
                                        token_categories={TokenCategory.full_set(), TokenCategory.Builder().add({KeyWord}).build()}),
                                )

    assert len(actual.scenarios) == 2
    scenario1 = actual.scenarios[EvaluationScenario('perplexity', TokenCategory.Builder().add({KeyWord}).build())]
    assert int(scenario1.sum) == 8
    assert scenario1.n_samples == 1
    assert scenario1.exp

    scenario2 = actual.scenarios[EvaluationScenario('perplexity', TokenCategory.full_set())]
    assert int(scenario2.sum) == 57
    assert scenario2.n_samples == 7
    assert scenario2.exp


def test_train():
    np.random.seed(13)
    torch.manual_seed(13)

    model = train(comet=False, device_options=DeviceOptions(fallback_to_cpu=True),
          training_config=LMTrainingConfig(Corpus(os.path.join(project_dir, 'data', 'dev')),
                                           training=Training(schedule=RafaelsTrainingSchedule(max_epochs=1))))
    assert int(model.metrics.bin_entropy) == 9
