import random
from collections import defaultdict
from pathlib import Path
from typing import List, Mapping, Any, Optional, Union, Callable, Dict

import pandas
from pandas import DataFrame, MultiIndex, Series

from langmodels.evaluation.characteristics import Characteristic
from langmodels.util.misc import merge_dicts_


def _to_sums(row: Series):
    dct = {EvaluationResultAccumulator.N_SAMPLES_COLUMN: row[EvaluationResultAccumulator.N_SAMPLES_COLUMN],
           EvaluationResultAccumulator.EXAMPLE_COLUMN: row[EvaluationResultAccumulator.EXAMPLE_COLUMN]}
    for name in row.keys():
        if name not in [EvaluationResultAccumulator.N_SAMPLES_COLUMN, EvaluationResultAccumulator.EXAMPLE_COLUMN]:
            dct[name] = row[name] * row[EvaluationResultAccumulator.N_SAMPLES_COLUMN]

    return Series(dct)


def _to_averages(row: Series) -> Series:
    dct = {EvaluationResultAccumulator.N_SAMPLES_COLUMN: row[EvaluationResultAccumulator.N_SAMPLES_COLUMN],
           EvaluationResultAccumulator.EXAMPLE_COLUMN: row[EvaluationResultAccumulator.EXAMPLE_COLUMN]}
    for name in row.keys():
        if name not in {EvaluationResultAccumulator.N_SAMPLES_COLUMN, EvaluationResultAccumulator.EXAMPLE_COLUMN}:
            dct[name] = row[name] / row[EvaluationResultAccumulator.N_SAMPLES_COLUMN]

    return Series(dct)


class EvaluationResult(object):
    def __init__(self, data: DataFrame):
        self.data = data

    def save(self, save_to: Path) -> None:
        self.data.to_html(save_to / f'{EvaluationResultAccumulator.FILE_NAME}.html')
        self.data.to_csv(save_to / f'{EvaluationResultAccumulator.FILE_NAME}.csv')
        self.data.to_pickle(save_to / f'{EvaluationResultAccumulator.FILE_NAME}.pkl')

    @classmethod
    def from_path(cls, path: Path) -> 'EvaluationResult':
        return cls.from_data(pandas.read_pickle(path /f'{EvaluationResultAccumulator.FILE_NAME}.pkl'))

    @classmethod
    def from_data(cls, dataframe: DataFrame) -> 'EvaluationResult':
        return cls(dataframe)

    def total(self) -> Mapping[str, Any]:
        dct = self.aggregate().data.loc[True].to_dict()
        del(dct[EvaluationResultAccumulator.EXAMPLE_COLUMN])
        return dct

    def aggregate(self, by: Optional[List[str]] = None) -> 'EvaluationResult':
        result_sum = self.data.apply(_to_sums, axis=1)
        return EvaluationResult(self._aggregate(result_sum, by).apply(_to_averages, axis=1))

    def _aggregate(self, data: DataFrame, groups: Optional[List[str]]) -> DataFrame:
        if groups is None or groups == [None]:
            groups = [True] * len(data)
        else:
            for group in groups:
                if group not in data.index.names:
                    raise ValueError(f'Characteristic does not exist: {group}. Must be one of: {[column for column in data.index.names]}')

        agg_dict = {column: (column, EvaluationResultAccumulator.metric_name_to_agg_function[column]) for column in data.columns}
        return data.groupby(groups).agg(**agg_dict)


class EvaluationResultAccumulator(object):
    """
    >>> def characterizer1(): pass
    >>> evaluation_result_acc = EvaluationResultAccumulator.empty([characterizer1], ['Entropy'])
    >>> evaluation_result_acc.add('Entropy', 'token1', ['Identifier'], 2.5)
    >>> evaluation_result_acc.add('Entropy', 'token2', ['Identifier'], 3.5)
    >>> evaluation_result_acc.add('Entropy', 'token3', ['KeyWord'], 12.)
    >>> dict(evaluation_result_acc._dct)
    {('Identifier',): {'Entropy': 6.0, 'n_samples': 2, 'example': 'token2'}, ('KeyWord',): {'Entropy': 12.0, 'n_samples': 1, 'example': 'token3'}}
    >>> evaluation_result = evaluation_result_acc.build()
    >>> evaluation_result.data # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          2  token2      3.0
    KeyWord             1  token3     12.0
    >>> evaluation_result.aggregate(['function']).data  # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          2  token2      3.0
    KeyWord             1  token3     12.0
    >>> evaluation_result.aggregate(['non_existing_characteristic'])
    Traceback (most recent call last):
    ...
    ValueError: Characteristic does not exist: non_existing_characteristic. Must be one of: ['function']
    >>> evaluation_result.total()
    {'n_samples': 3, 'Entropy': 6.0}
    >>> dict(evaluation_result_acc.merge(evaluation_result_acc)._dct) # doctest: +NORMALIZE_WHITESPACE
    {('Identifier',): {'Entropy': 12.0, 'n_samples': 4, 'example': 'token2'}, ('KeyWord',): {'Entropy': 24.0, 'n_samples': 2, 'example': 'token3'}}
    >>> import tempfile
    >>> d = tempfile.TemporaryDirectory()
    >>> evaluation_result.save(Path(d.name))
    >>> EvaluationResult.from_path(Path(d.name)).data # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          2  token2     3.0
    KeyWord             1  token3     12.0

    """

    N_SAMPLES_COLUMN = 'n_samples'
    EXAMPLE_COLUMN = 'example'
    FILE_NAME = 'evaluation'

    metric_name_to_agg_function: Mapping[str, Union[str, Callable]] = {'Entropy': sum, N_SAMPLES_COLUMN: sum,
                                                                       EXAMPLE_COLUMN: lambda lst: random.sample(list(lst), 1)[0]}
    metric_name_to_init_value: Mapping[str, Any] = {'Entropy': 0., N_SAMPLES_COLUMN: 0, EXAMPLE_COLUMN: None}

    def __init__(self, dct: Dict, characteristics: List[Characteristic], metric_names: List[str]):
        self._dct = dct
        self.characteristics = characteristics
        self.metric_names = metric_names

    @classmethod
    def empty(cls, characteristics: List[Characteristic], metric_names: List[str]) -> 'EvaluationResultAccumulator':
        metric_names = metric_names + [EvaluationResultAccumulator.N_SAMPLES_COLUMN, EvaluationResultAccumulator.EXAMPLE_COLUMN]

        def empty_result(metric_names: List[str]) -> Mapping[str, Any]:
            return {metric_name: EvaluationResultAccumulator.metric_name_to_init_value[metric_name]
                    for metric_name in metric_names}

        return cls(defaultdict(lambda: empty_result(metric_names)), characteristics, metric_names)

    def add(self, metric_name: str, token: str, token_characteristics: List[Any], metric_value: float) -> None:
        prev_metrics = self._dct[tuple(token_characteristics)]
        prev_metrics[EvaluationResultAccumulator.N_SAMPLES_COLUMN] += 1
        prev_metrics[EvaluationResultAccumulator.EXAMPLE_COLUMN] = token
        prev_metrics[metric_name] += metric_value
        self._dct[tuple(token_characteristics)] = prev_metrics

    def merge(self, other: 'EvaluationResultAccumulator') -> 'EvaluationResultAccumulator':
        def value_merger(metric_values1, metric_values2):
            result = {}
            for metric_name in self.metric_names:
                result[metric_name] = EvaluationResultAccumulator.metric_name_to_agg_function[metric_name]([metric_values1[metric_name], metric_values2[metric_name]])

            return result

        return EvaluationResultAccumulator(merge_dicts_(self._dct, other._dct, value_merger), self.characteristics, self.metric_names)

    def build(self) -> EvaluationResult:
        dataframe_index = MultiIndex.from_tuples(self._dct.keys(),
                                                  names=list(map(lambda f: type(f).__name__, self.characteristics)))

        return EvaluationResult(DataFrame(self._dct.values(), columns=self.metric_names, index=dataframe_index)
                                .sort_index().apply(_to_averages, axis=1))

