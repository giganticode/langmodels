from unittest.mock import Mock

from dataprep.tokens.containers import SplitContainer, OneLineComment
from langmodels.evaluation.filtering import TokenTypeSubset, EvaluationCustomization
from langmodels.evaluation.metrics import bin_entropy, mrr, EvaluationResult
from langmodels.model import TrainedModel

any_1 = 'java'


def test_bin_entropy_empty():
    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.get_entropies_for_text.return_value = ([], [], [])

    expected = {EvaluationCustomization.no_customization(): EvaluationResult([], [], 0.)}
    actual = bin_entropy(trained_model_mock, '', extension=any_1, append_eof=False, )

    assert expected == actual


def test_bin_entropy_simple_args():
    trained_model_mock = Mock(spec=TrainedModel)
    entropies = [1.0, 2.0]
    prep_text = ['My', 'Class</t>']
    trained_model_mock.get_entropies_for_text.return_value = (entropies, prep_text, [SplitContainer, SplitContainer])
    token_set = TokenTypeSubset.Builder().add(SplitContainer).build()

    expected = {EvaluationCustomization(token_set): EvaluationResult(prep_text, entropies, 1.5)}
    actual = bin_entropy(trained_model_mock, 'MyClass', extension=any_1, append_eof=False,
                         evaluation_customizations={EvaluationCustomization(token_set)})

    assert expected == actual


def test_bin_entropy_with_comment():
    trained_model_mock = Mock(spec=TrainedModel)
    prep_text = ['My', 'Class</t>', '/', '/']
    trained_model_mock.get_entropies_for_text.return_value = (
        [1.0, 2.0, 3.0, 6.0],
        prep_text,
        [SplitContainer, SplitContainer, OneLineComment, OneLineComment]
    )

    expected = {
        EvaluationCustomization(TokenTypeSubset.full_set()):
            EvaluationResult(prep_text, [1.0, 2.0, 3.0, 6.0], 3.0),

        EvaluationCustomization(TokenTypeSubset.only_comments()):
            EvaluationResult(prep_text, [None, None, 3.0, 6.0], 4.5),

        EvaluationCustomization(TokenTypeSubset.full_set_without_comments()):
            EvaluationResult(prep_text, [1.0, 2.0, None, None], 1.5)
    }

    actual = bin_entropy(trained_model_mock, 'MyClass //', extension='java', append_eof=False,
                         evaluation_customizations={
                             EvaluationCustomization(TokenTypeSubset.full_set()),
                             EvaluationCustomization(TokenTypeSubset.only_comments()),
                             EvaluationCustomization(TokenTypeSubset.full_set_without_comments())
                         }, full_tokens=False)

    assert expected == actual


def test_mrr_default_args():
    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.get_predictions_and_feed.side_effect = [
        [([('a1</t>', 0.), ('b1</t>', 0.)], 'a1</t>', SplitContainer),
         ([('a2</t>', 0.), ('b2</t>', 0.)], 'b2</t>', SplitContainer)]
    ]

    expected = {
        EvaluationCustomization.no_customization():
            EvaluationResult(['a1</t>', 'b2</t>'], [1.0, 0.5], 0.75)
    }

    actual = mrr(trained_model_mock, 'a1 b2', extension='java', append_eof=False,
                 evaluation_customizations={EvaluationCustomization.no_customization()})

    assert actual == expected


def test_mrr_default_all_token_types():
    trained_model_mock = Mock(spec=TrainedModel)
    prep_tokens = ['a1</t>', 'b2</t>', '/</t>', '/</t>']
    method_call_result = [([('a1</t>', 0.), ('b1</t>', 0.)], prep_tokens[0], SplitContainer),
                                ([('a2</t>', 0.), ('b2</t>', 0.)], prep_tokens[1], SplitContainer),
                                ([('a3</t>', 0.), ('b3</t>', 0.)], prep_tokens[2], OneLineComment),
                                ([('a4</t>', 0.), ('b4</t>', 0.)], prep_tokens[3], OneLineComment)]

    trained_model_mock.get_predictions_and_feed.side_effect = [method_call_result] * 3

    expected = {
        EvaluationCustomization(TokenTypeSubset.full_set()):
            EvaluationResult(prep_tokens, [1.0, None, 0.5, None, 0., 0.], 0.375),
        EvaluationCustomization(TokenTypeSubset.only_comments()):
            EvaluationResult(prep_tokens, [None, None, None, None, 0., 0.], 0.),
        EvaluationCustomization(TokenTypeSubset.full_set_without_comments()):
            EvaluationResult(prep_tokens, [1.0, None, 0.5, None, None, None], 0.75)
    }
    actual = mrr(trained_model_mock, 'a1 b2 //', 'java', append_eof=False,
                 evaluation_customizations={
                     EvaluationCustomization(TokenTypeSubset.full_set()),
                     EvaluationCustomization(TokenTypeSubset.only_comments()),
                     EvaluationCustomization(TokenTypeSubset.full_set_without_comments())
                 })

    assert sorted(actual, key=lambda s: str(s)) == sorted(expected, key=lambda s: str(s))

