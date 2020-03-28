from typing import List

from columnar import columnar

from langmodels.model import ModelDescription


def query_all_models(cached: bool = False) -> List[ModelDescription]:
    return _get_all_models_query(cached=cached).sorted_by_entropy().execute()


class _ModelQuery(object):
    def __init__(self, previous_query: '_ModelQuery'):
        self.previous_query = previous_query

    def __str__(self):
        desc_list = self.execute()
        return columnar(headers=ModelDescription.get_attribute_list(),
                        data=list(map(lambda l: l.get_value_list(), desc_list)),
                        no_borders=True, terminal_width=200)

    def sorted_by_entropy(self) -> '_ModelQuery':
        return _SortByEntropyQuery(self)

    def get_previous_query(self) -> '_ModelQuery':
        return self.previous_query


def _get_all_models_query(cached: bool) -> _ModelQuery:
    return _GetAllModelsQuery(cached=cached)


class _GetAllModelsQuery(_ModelQuery):
    def __init__(self, cached: bool):
        super().__init__(None)
        self.cached = cached


class _SortByEntropyQuery(_ModelQuery):
    def __init__(self, previous_query: _ModelQuery):
        super().__init__(previous_query)

    def execute(self) -> List[ModelDescription]:
        return sorted(self.get_previous_query().execute(), key=lambda m: m.bin_entropy)