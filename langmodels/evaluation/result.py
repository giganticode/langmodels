import random
from pathlib import Path
from typing import List, Mapping, Any, Optional, Union, Callable

import pandas
from pandas import DataFrame, MultiIndex, Series

from langmodels.evaluation.characteristics import Characteristic


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

    @staticmethod
    def to_sum(row: Series):
        dct = {EvaluationResultAccumulator.N_SAMPLES_COLUMN: row[EvaluationResultAccumulator.N_SAMPLES_COLUMN],
               EvaluationResultAccumulator.EXAMPLE_COLUMN: row[EvaluationResultAccumulator.EXAMPLE_COLUMN]}
        for name in row.keys():
            if name not in [EvaluationResultAccumulator.N_SAMPLES_COLUMN, EvaluationResultAccumulator.EXAMPLE_COLUMN]:
                dct[name] = row[name] * row[EvaluationResultAccumulator.N_SAMPLES_COLUMN]

        return Series(dct)

    def to_accumulator(self) -> 'EvaluationResultAccumulator':
        return EvaluationResultAccumulator(self.data.apply(EvaluationResult.to_sum, axis=1))

    def total(self) -> Mapping[str, Any]:
        dct = self.aggregate().data.loc[True].to_dict()
        del(dct[EvaluationResultAccumulator.EXAMPLE_COLUMN])
        return dct

    def aggregate(self, by: Optional[List[str]]=None) -> 'EvaluationResult':
        return self.to_accumulator().sum(by).build()


class EvaluationResultAccumulator(object):
    """
    >>> def characterizer1(): pass
    >>> evaluation_result_acc = EvaluationResultAccumulator.empty([characterizer1], ['Entropy'])
    >>> evaluation_result_acc.add('Entropy', 'token1', ['Identifier'], 2.5)
    >>> evaluation_result_acc.add('Entropy', 'token2', ['Identifier'], 3.5)
    >>> evaluation_result_acc.add('Entropy', 'token3', ['KeyWord'], 12.)
    >>> evaluation_result_acc._data # doctest: +NORMALIZE_WHITESPACE
               n_samples Entropy example
    function
    Identifier         2       6  token1
    KeyWord            1      12  token3
    >>> evaluation_result = evaluation_result_acc.build()
    >>> evaluation_result.data # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          2  token1      3.0
    KeyWord             1  token3     12.0
    >>> evaluation_result.to_accumulator()._data  # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          2  token1      6.0
    KeyWord             1  token3     12.0
    >>> evaluation_result.aggregate(['function']).data  # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          2  token1      3.0
    KeyWord             1  token3     12.0
    >>> evaluation_result.aggregate(['non_existing_characteristic'])
    Traceback (most recent call last):
    ...
    ValueError: Characteristic does not exist: non_existing_characteristic. Must be one of: ['function']
    >>> evaluation_result.total()
    {'n_samples': 3, 'Entropy': 6.0}
    >>> evaluation_result_acc.merge(evaluation_result_acc)._data # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          4  token1     12.0
    KeyWord             2  token3     24.0
    >>> import tempfile
    >>> d = tempfile.TemporaryDirectory()
    >>> evaluation_result.save(Path(d.name))
    >>> EvaluationResult.from_path(Path(d.name)).data # doctest: +NORMALIZE_WHITESPACE
                n_samples example  Entropy
    function
    Identifier          2  token1      3.0
    KeyWord             1  token3     12.0

    """

    N_SAMPLES_COLUMN = 'n_samples'
    EXAMPLE_COLUMN = 'example'
    FILE_NAME = 'evaluation'

    metric_name_to_agg_function: Mapping[str, Union[str, Callable]] = {'Entropy': 'sum', 'n_samples': 'sum',
                                                                       EXAMPLE_COLUMN: lambda lst: random.sample(list(lst), 1)[0]}

    def __init__(self, data: DataFrame):
        self._data = data

    def sum(self, groups: List[str]) -> 'EvaluationResultAccumulator':
        data = self._data
        if groups:
            for group in groups:
                if group not in data.index.names:
                    raise ValueError(f'Characteristic does not exist: {group}. Must be one of: {[column for column in data.index.names]}')
        else:
            groups = [True] * len(data)
        agg_dict = {column: (column, EvaluationResultAccumulator.metric_name_to_agg_function[column]) for column in data.columns}
        return EvaluationResultAccumulator(data.groupby(groups).agg(**agg_dict))

    @classmethod
    def empty(cls, characterizers: List[Characteristic], metrics: List[str]) -> 'EvaluationResultAccumulator':
        dataframe_columns = [EvaluationResultAccumulator.N_SAMPLES_COLUMN] + metrics
        dataframe_index = MultiIndex.from_product(iterables=[[]] * len(characterizers),
                                                  names=list(map(lambda f: type(f).__name__, characterizers)))

        return cls(DataFrame(columns=dataframe_columns, index=dataframe_index))

    def add(self, metric_name: str, token: str, token_characteristics: List[Any], entropy: float) -> None:
        try:
            self._data.loc[tuple(token_characteristics), EvaluationResultAccumulator.N_SAMPLES_COLUMN] += 1
        except:
            self._data.loc[tuple(token_characteristics), EvaluationResultAccumulator.N_SAMPLES_COLUMN] = 1
            self._data.loc[tuple(token_characteristics), metric_name] = 0
            self._data.loc[tuple(token_characteristics), EvaluationResultAccumulator.EXAMPLE_COLUMN] = token
        self._data.loc[tuple(token_characteristics), metric_name] += entropy
        if random.randint(0, 9) == 0:
            self._data.loc[tuple(token_characteristics), EvaluationResultAccumulator.EXAMPLE_COLUMN] = token

    def merge(self, other: 'EvaluationResultAccumulator') -> 'EvaluationResultAccumulator':
        return EvaluationResultAccumulator(self._data.append(other._data))\
            .build().aggregate(self._data.index.names).to_accumulator()
        # return EvaluationResultAccumulator(self._data.add(other._data, fill_value=0))

    @staticmethod
    def to_averages(row: Series) -> Series:
        dct = {EvaluationResultAccumulator.N_SAMPLES_COLUMN: row[EvaluationResultAccumulator.N_SAMPLES_COLUMN],
               EvaluationResultAccumulator.EXAMPLE_COLUMN: row[EvaluationResultAccumulator.EXAMPLE_COLUMN]}
        for name in row.keys():
            if name not in {EvaluationResultAccumulator.N_SAMPLES_COLUMN, EvaluationResultAccumulator.EXAMPLE_COLUMN}:
                dct[name] = row[name] / row[EvaluationResultAccumulator.N_SAMPLES_COLUMN]

        return Series(dct)

    def build(self) -> EvaluationResult:
        return EvaluationResult.from_data(self._data
                                          .sort_index()
                                          .apply(EvaluationResultAccumulator.to_averages, axis=1))

