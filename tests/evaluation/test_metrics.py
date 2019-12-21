from unittest.mock import Mock

from dataprep.preprocess.metadata import PreprocessingMetadata
from dataprep.tokens.containers import SplitContainer, OneLineComment
from langmodels.evaluation.filtering import TokenTypeSubset, FullTokenIterator, \
    SubtokenIterator
from langmodels.evaluation.metrics import bin_entropy, mrr, EvaluationResult, EvaluationScenario
from langmodels.model import TrainedModel


def test_bin_entropy_empty():
    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.get_entropies_for_prep_text.return_value = []

    expected = {EvaluationScenario('full_token_entropy', TokenTypeSubset.full_set()): EvaluationResult([], 0.0, 0)}
    actual = bin_entropy(trained_model_mock, [], PreprocessingMetadata())

    assert expected == actual


def test_bin_entropy_simple_args():
    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.get_entropies_for_prep_text.return_value = [1.0, 2.0]

    expected = {EvaluationScenario('full_token_entropy', TokenTypeSubset.full_set()): EvaluationResult([1.0, 2.0], 3.0, 1)}
    actual = bin_entropy(trained_model_mock, ['My', 'Class</t>'],
                         PreprocessingMetadata(word_boundaries=[0, 2], token_types=[SplitContainer]))

    assert expected == actual


def test_bin_entropy_with_comment():
    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.get_entropies_for_prep_text.return_value = [1.0, 2.0, 3.0, 6.0]

    expected = {
        EvaluationScenario('full_token_entropy', TokenTypeSubset.full_set()):
            EvaluationResult([1.0, 2.0, 3.0, 6.0], 4.0, 3),

        EvaluationScenario('full_token_entropy', TokenTypeSubset.only_comments()):
            EvaluationResult([None, None, 3.0, 6.0], 4.5, 2),

        EvaluationScenario('full_token_entropy', TokenTypeSubset.full_set_without_comments()):
            EvaluationResult([1.0, 2.0, None, None], 3.0, 1),

        EvaluationScenario('subtoken_entropy', TokenTypeSubset.full_set()):
            EvaluationResult([1.0, 2.0, 3.0, 6.0], 3.0, 4),

        EvaluationScenario('subtoken_entropy', TokenTypeSubset.only_comments()):
            EvaluationResult([None, None, 3.0, 6.0], 4.5, 2),

        EvaluationScenario('subtoken_entropy', TokenTypeSubset.full_set_without_comments()):
            EvaluationResult([1.0, 2.0, None, None], 1.5, 2)
    }

    actual = bin_entropy(trained_model_mock, ['My', 'Class</t>', '/', '/'],
                         PreprocessingMetadata(word_boundaries=[0, 2, 3, 4],
                                               token_types=[SplitContainer, OneLineComment, OneLineComment]),
                         token_types_subsets={
                             TokenTypeSubset.full_set(),
                             TokenTypeSubset.only_comments(),
                             TokenTypeSubset.full_set_without_comments()
                         },
                         token_iterator_types={FullTokenIterator, SubtokenIterator})

    assert expected == actual


def test_mrr_default_args():
    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.predict_next_full_token.side_effect = [[('a1', 0.), ('b1', 0.)],
                                                              [('a2', 0.), ('b2', 0.)]]

    expected = {EvaluationScenario('mrr', TokenTypeSubset.full_set()): EvaluationResult([1.0, 0.5], 0.75, 2)}
    actual = mrr(trained_model_mock, ['a1</t>', 'b2</t>'],
                 metadata=PreprocessingMetadata(word_boundaries=[0, 1, 2],
                                                token_types=[SplitContainer, SplitContainer]))

    assert actual == expected


def test_mrr_default_all_token_types():
    trained_model_mock = Mock(spec=TrainedModel)
    trained_model_mock.predict_next_full_token.side_effect = [[('a1', 0.), ('b1', 0.)],
                                                              [('a2', 0.), ('b2', 0.)],
                                                              [('a3', 0.), ('b3', 0.)],
                                                              [('a4', 0.), ('b4', 0.)],

                                                              [('a1', 0.), ('b1', 0.)],
                                                              [('a2', 0.), ('b2', 0.)],
                                                              [('a3', 0.), ('b3', 0.)],
                                                              [('a4', 0.), ('b4', 0.)],

                                                              [('a1', 0.), ('b1', 0.)],
                                                              [('a2', 0.), ('b2', 0.)],
                                                              [('a3', 0.), ('b3', 0.)],
                                                              [('a4', 0.), ('b4', 0.)]
                                                              ]

    expected = {
        EvaluationScenario('mrr', TokenTypeSubset.full_set()):
            EvaluationResult([1.0, None, 0.5, None, 0., 0.], 0.375, 4),
        EvaluationScenario('mrr', TokenTypeSubset.only_comments()):
            EvaluationResult([None, None, None, None, 0., 0.], 0., 2),
        EvaluationScenario('mrr', TokenTypeSubset.full_set_without_comments()):
            EvaluationResult([1.0, None, 0.5, None, None, None], 0.75, 2)
    }
    actual = mrr(trained_model_mock, ['a', '1</t>', 'b', '2</t>', '/</t>', '/</t>'],
                 metadata=PreprocessingMetadata(word_boundaries=[0, 2, 4, 5, 6],
                                                token_types=[SplitContainer, SplitContainer, OneLineComment, OneLineComment]),
                 token_types_subsets={
                     TokenTypeSubset.full_set(),
                     TokenTypeSubset.only_comments(),
                     TokenTypeSubset.full_set_without_comments()
                 })

    assert sorted(actual, key=lambda s: str(s)) == sorted(expected, key=lambda s: str(s))

