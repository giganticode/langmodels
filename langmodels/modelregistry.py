import hashlib
import logging
import os

import requests
from columnar import columnar
from typing import List

from langmodels import MODEL_ZOO_PATH
from langmodels.model import TrainedModel, ModelDescription

logger = logging.getLogger(__name__)

MODEL_DIR_URL = 'https://www.inf.unibz.it/~hbabii/pretrained_models'
MODEL_LIST_URL = MODEL_DIR_URL + '.list'


def _download_file(url: str, path: str, check_md5: bool = False):
    response = requests.get(url, stream=True)
    with open(path + '.tmp', "wb") as f:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)

    if check_md5:
        correct_md5 = requests.get(url + ".md5").content.decode(encoding='utf8').rstrip('\n')
        md5 = hashlib.md5(open(path + '.tmp', 'rb').read()).hexdigest()
        if md5 != correct_md5:
            logger.warning(f'The md5 checksum for {url} is not correct: {md5}. Downloading the file again...')
            _download_file(url, path, check_md5=True)

    os.replace(path + '.tmp', path)


def _get_all_model_ids(cached: bool = False) -> List[str]:
    if cached:
        root, dirs, files = next(os.walk(MODEL_ZOO_PATH))
        return [d for d in dirs if os.path.exists(os.path.join(root, d, 'best.pth'))]
    else:
        content = requests.get(MODEL_LIST_URL).content.decode(encoding='utf8')
        model_list = content.rstrip('\n').split('\n')
        return model_list


def _get_all_models(cached: bool):
    model_ids = _get_all_model_ids(cached=cached)
    loaded_models = [load_model_by_id(model_id, load_description_only=True).get_model_description()
                     for model_id in model_ids]
    return loaded_models


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


class _SortByEntropyQuery(_ModelQuery):
    def __init__(self, previous_query: _ModelQuery):
        super().__init__(previous_query)

    def execute(self) -> List[ModelDescription]:
        return sorted(self.get_previous_query().execute(), key=lambda m: m.bin_entropy)


class _GetAllModelsQuery(_ModelQuery):
    def __init__(self, cached: bool):
        super().__init__(None)
        self.cached = cached

    def execute(self) -> List[ModelDescription]:
        return _get_all_models(cached=self.cached)


def _get_all_models_query(cached: bool) -> _ModelQuery:
    return _GetAllModelsQuery(cached=cached)


def query_all_models(cached: bool = False) -> List[ModelDescription]:
    return _get_all_models_query(cached=cached).sorted_by_entropy().execute()


def list_pretrained_models(cached: bool = False) -> None:
    print(_get_all_models_query(cached=cached).sorted_by_entropy())


def load_from_path(path: str, force_use_cpu: bool = False, load_description_only: bool = False) -> TrainedModel:
    return TrainedModel(path, force_use_cpu, load_description_only)


def load_model_by_id(id: str, force_use_cpu: bool = False, load_description_only: bool = False) -> TrainedModel:
    path = os.path.join(MODEL_ZOO_PATH, id)
    if not os.path.exists(os.path.join(path, 'best.pth')):
        if id not in _get_all_model_ids():
            raise ValueError(f'Model with id {id} is not found on the server.')
        url_to_model_dir = MODEL_DIR_URL + '/' + id
        if not os.path.exists(path):
            os.makedirs(path)
        _download_file(url_to_model_dir + '/config', os.path.join(path, 'config'))
        _download_file(url_to_model_dir + '/metrics', os.path.join(path, 'metrics'))
        _download_file(url_to_model_dir + '/tags', os.path.join(path, 'tags'))
        if not load_description_only:
            logger.info(f'Model is not found in cache. Downloading from {url_to_model_dir} ...')
            _download_file(url_to_model_dir + '/best.pth', os.path.join(path, 'best.pth'), check_md5=True)
            _download_file(url_to_model_dir + '/vocab', os.path.join(path, 'vocab'), check_md5=True)

    return load_from_path(path, force_use_cpu, load_description_only)


def load_model_with_tag(tag: str, force_use_cpu: bool = False):
    for cached in [True, False]:  # first checking tags of cached models
        for model in query_all_models(cached=cached):
            if model.is_tagged_by(tag):
                return load_model_by_id(model.id, force_use_cpu=force_use_cpu)
    raise ValueError(f'Model tagged with {tag} not found.')


def load_default_model(force_use_cpu: bool = False) -> TrainedModel:
    return load_model_with_tag('DEFAULT', force_use_cpu=force_use_cpu)


__all__ = [load_default_model, list_pretrained_models, query_all_models, load_model_by_id, load_model_with_tag, load_from_path]
