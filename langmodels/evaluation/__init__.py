import os

from langmodels.evaluation.api import evaluate_on_file, evaluate_on_string, evaluate_on_path
from langmodels.evaluation.characteristics import SubtokenNumber, TokenType, FrequencyRank, Project, LinePosition, \
    TokenPosition
from langmodels.evaluation.options import EvaluationOptions
from langmodels.evaluation.result import EvaluationResult

__all__ = [
    'evaluate_on_file',
    'evaluate_on_string',
    'evaluate_on_path',
    'EvaluationResult',
    'EvaluationOptions',
    'SubtokenNumber',
    'TokenType',
    'FrequencyRank',
    'Project',
    'LinePosition',
    'TokenPosition'
]


if 'LANGMODELS_EVALUATION_TEST' in os.environ:
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)


