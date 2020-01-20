from unittest import mock
from unittest.mock import MagicMock, Mock

from pytest_mock import MockFixture

from dataprep.tokens.containers import SplitContainer
from langmodels.evaluation.evaluation import evaluate_model_on_string
from langmodels.evaluation.customization import TokenTypeSubset
from langmodels.evaluation.metrics import EvaluationResult, EvaluationScenario, Evaluation
from langmodels.model import TrainedModel


def test_evaluate_model_on_string_empty():
    trained_model_mock = MagicMock(spec=TrainedModel)
    trained_model_mock.get_entropies_for_text.return_value = ([], [], [])

    expected = [Evaluation(
        '',
        {EvaluationScenario('full_token_entropy'): EvaluationResult([], [], [], 0.)}
    )]
    actual = evaluate_model_on_string(trained_model_mock, '')

    assert actual == expected


def test_evaluate_on_string_default_args(mocker: MockFixture):
    text = 'MyClass'
    prep_line = ['My', 'Class</t>']
    types = [SplitContainer, SplitContainer]
    result = EvaluationResult(prep_line, list(map(lambda t: t.__name__, types)), [1.0, 2.0], 3.0)
    scenarios = {EvaluationScenario('full_token_entropy'): result}

    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.get_entropies_for_text.return_value = ([1.0, 2.0], prep_line, types)

    mocked_metric = Mock(spec=callable, return_value={TokenTypeSubset.full_set(): result})
    mocker.patch('langmodels.evaluation.evaluation._get_metric_by_name', new=lambda x: mocked_metric)
    mocker.patch('langmodels.evaluation.evaluation.get_metrics_name', new=lambda x: 'full_token_entropy')

    actual = evaluate_model_on_string(trained_model_mock, text)

    mocked_metric.assert_called_with(trained_model_mock, text, 'java', False, None)
    assert actual == [Evaluation(text, scenarios)]


def test_evaluate_on_string_default_args_not_result_per_line(mocker: MockFixture):
    # given
    text = 'MyClass\n{'
    result = Mock(spec=EvaluationResult)
    scenarios = {EvaluationScenario('full_token_entropy'): result}

    trained_model_mock = Mock(spec=TrainedModel)

    mocked_metric = Mock(spec=callable, return_value={TokenTypeSubset.full_set(): result})
    mocker.patch('langmodels.evaluation.evaluation._get_metric_by_name', new=lambda x: mocked_metric)
    mocker.patch('langmodels.evaluation.evaluation.get_metrics_name', new=lambda x: 'full_token_entropy')

    # when
    actual = evaluate_model_on_string(trained_model_mock, text, result_per_line=False)

    # then
    mocked_metric.assert_called_with(trained_model_mock, text, 'java', False, None)
    assert actual == Evaluation(text, scenarios)


def test_evaluate_on_string_non_default_token_types_and_metrics_multiline(mocker: MockFixture):
    text = 'MyClass\n{'
    token_type_subsets = {TokenTypeSubset.full_set(), TokenTypeSubset.full_set_without_comments()}

    metrics = {'full_token_entropy', 'mrr'}
    scenarios = {EvaluationScenario(metric, token_type_subset)
                   for token_type_subset in token_type_subsets for metric in metrics}

    trained_model_mock = Mock(spec=TrainedModel)

    evaluation_mocks = [Mock(spec=Evaluation)] * 2
    mocked_evaluate_on_line = Mock(spec=callable)
    mocked_evaluate_on_line.side_effect = evaluation_mocks
    mocker.patch('langmodels.evaluation.evaluation._evaluate_model_on_line', new=mocked_evaluate_on_line)

    actual = evaluate_model_on_string(trained_model_mock, text, 'java', metrics, token_type_subsets,
                                      result_per_line=True, append_eof=True)

    mocked_evaluate_on_line.assert_has_calls([
        mock.call(trained_model_mock, 'MyClass', 'java', metrics, token_type_subsets, False),
        mock.call(trained_model_mock, '{', 'java', metrics, token_type_subsets, True)
    ])

    assert actual == evaluation_mocks
