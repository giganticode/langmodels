from unittest import mock

from pytest_mock import MockFixture

from dataprep.parse.model.metadata import PreprocessingMetadata
from unittest.mock import MagicMock, Mock

from langmodels.evaluation import evaluate_model_on_string
from langmodels.evaluation.filtering import TokenTypes
from langmodels.evaluation.metrics import EvaluationResult, EvaluationScenario, Evaluation
from langmodels.model import TrainedModel


def test_evaluate_model_on_string_empty():
    trained_model_mock = MagicMock(spec=TrainedModel)
    trained_model_mock.prep_text.return_value = ([], PreprocessingMetadata())

    expected = [Evaluation(
        '', [], PreprocessingMetadata(),
        {EvaluationScenario('full_token_entropy', TokenTypes.ALL): EvaluationResult([], 0., 0)}
    )]
    actual = evaluate_model_on_string(trained_model_mock, '')

    assert actual == expected


def test_evaluate_on_string_default_args(mocker: MockFixture):
    # given
    text = 'MyClass'
    prep_line = ['My', 'Class</t>']
    metadata = PreprocessingMetadata(word_boundaries=[0, 2])
    scenarios = {EvaluationScenario('full_token_entropy', TokenTypes.ALL): EvaluationResult([1.0, 2.0], 3.0, 1)}

    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.prep_text.return_value = (prep_line, metadata)

    mocked_metric = Mock(spec=callable, return_value=scenarios)
    mocker.patch('langmodels.evaluation.evaluation.DEFAULT_METRIC', new=mocked_metric)

    # when
    actual = evaluate_model_on_string(trained_model_mock, text)

    # then
    trained_model_mock.prep_text.assert_has_calls([mock.call('MyClass', extension='java',
                                                             force_reinit_bpe_data=False,
                                                             return_metadata=True, append_eof=False)])
    mocked_metric.assert_called_with(trained_model_mock, prep_line, metadata, None)
    assert actual == [Evaluation(text, prep_line, metadata, scenarios)]


def test_evaluate_on_string_default_args_not_result_per_line(mocker: MockFixture):
    # given
    text = 'MyClass\n{'
    prep_line = ['My', 'Class</t>']
    metadata = Mock(spec=PreprocessingMetadata)
    scenarios = {EvaluationScenario('full_token_entropy', TokenTypes.ALL): Mock(spec=EvaluationResult)}

    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.prep_text.return_value = (prep_line, metadata)

    mocked_metric = Mock(spec=callable, return_value=scenarios)
    mocker.patch('langmodels.evaluation.evaluation.DEFAULT_METRIC', new=mocked_metric)

    # when
    actual = evaluate_model_on_string(trained_model_mock, text, result_per_line=False)

    # then
    trained_model_mock.prep_text.assert_has_calls([mock.call('MyClass\n{', extension='java',
                                                             force_reinit_bpe_data=False,
                                                             return_metadata=True, append_eof=False)])
    mocked_metric.assert_called_with(trained_model_mock, prep_line, metadata, None)
    assert actual == [Evaluation(text, prep_line, metadata, scenarios)]


def test_evaluate_on_string_non_default_token_types_and_metrics_multiline(mocker: MockFixture):
    # given
    text = 'MyClass\n{'
    prep_lines = [['My', 'Class</t>'], ['{']]
    metadata_list = [Mock(spec=PreprocessingMetadata) for i in range(len(prep_lines))]
    metrics = {'full_token_entropy', 'mrr'}
    token_types_list = {TokenTypes.ALL, TokenTypes.ALL_BUT_COMMENTS}
    scenarios = [[{EvaluationScenario(metric, token_types): Mock(spec=EvaluationResult)
                   for token_types in token_types_list} for i in range(len(prep_lines))] for metric in metrics]
    mocked_metrics = [Mock(spec=callable, side_effect=scenario) for scenario in scenarios]

    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.prep_text.side_effect = list(zip(prep_lines, metadata_list))

    mocked_metrics_from_strings = Mock(spec=callable, return_value=mocked_metrics)
    mocker.patch('langmodels.evaluation.evaluation.metrics_from_strings', new=mocked_metrics_from_strings)

    # when
    actual = evaluate_model_on_string(trained_model_mock, text,
                                      token_types=token_types_list,
                                      metrics=metrics,
                                      result_per_line=True, append_eof=True)

    # then
    trained_model_mock.prep_text.assert_has_calls([mock.call('MyClass', extension='java',
                                                             force_reinit_bpe_data=False,
                                                             return_metadata=True, append_eof=False)])
    trained_model_mock.prep_text.assert_has_calls([mock.call('{', extension='java',
                                                             force_reinit_bpe_data=False,
                                                             return_metadata=True, append_eof=True)])

    for mocked_metric in mocked_metrics:
        for prep_line, metadata in zip(prep_lines, metadata_list):
            mocked_metric.assert_has_calls([mock.call(trained_model_mock, prep_line, metadata, token_types_list)])

    assert actual == [Evaluation('MyClass', prep_lines[0], metadata_list[0],
                                 {k: v for ss in [scenarios[0][0], scenarios[1][0]] for k, v in ss.items()}),
                      Evaluation('{', prep_lines[1], metadata_list[1],
                                 {k: v for ss in [scenarios[0][1], scenarios[1][1]] for k, v in ss.items()})]
