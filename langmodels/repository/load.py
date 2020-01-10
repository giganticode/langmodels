import logging
import os
from threading import Lock
from typing import List

import requests

from langmodels import MODEL_ZOO_PATH
from langmodels.lmconfig.datamodel import CONFIG_VERSION
from langmodels.model import TrainedModel, METRICS_FILE_NAME, VOCAB_FILE_NAME, CONFIG_FILE_NAME, TAGS_FILE_NAME
from langmodels.repository.download import download_metadata, download_model
from langmodels.repository.query import query_all_models, _get_all_models_query, _GetAllModelsQuery
from langmodels.repository.settings import MODEL_LIST_URL, MODEL_DIR_URL
from langmodels.model import BEST_MODEL_FILE_NAME

logger = logging.getLogger(__name__)


def _get_all_model_ids(cached: bool = False) -> List[str]:
    if not os.path.exists(MODEL_ZOO_PATH):
        os.makedirs(MODEL_ZOO_PATH)

    if cached:
        root, dirs, files = next(os.walk(MODEL_ZOO_PATH))
        return [d for d in dirs if os.path.exists(os.path.join(root, d, BEST_MODEL_FILE_NAME))]
    else:
        content = requests.get(MODEL_LIST_URL).content.decode(encoding='utf8')
        model_list = content.rstrip('\n').split('\n')
        return model_list


def _get_all_models(cached: bool) -> List[TrainedModel]:
    model_ids = _get_all_model_ids(cached=cached)
    loaded_models = [load_model_by_id(model_id, load_description_only=True).get_model_description()
                     for model_id in model_ids]
    return loaded_models


_GetAllModelsQuery.execute = lambda self: _get_all_models(cached=self.cached)


def list_pretrained_models(cached: bool = False) -> None:
    print(_get_all_models_query(cached=cached).sorted_by_entropy())


def load_from_path(path: str, force_use_cpu: bool = False, load_description_only: bool = False) -> TrainedModel:
    return TrainedModel(path, force_use_cpu, load_description_only)


MODEL_DATA_FILES = [BEST_MODEL_FILE_NAME, VOCAB_FILE_NAME]
MODEL_METADATA_FILES = [CONFIG_FILE_NAME, METRICS_FILE_NAME, TAGS_FILE_NAME]

load_model_lock = Lock()


def load_model_by_id(id: str, force_use_cpu: bool = False, load_description_only: bool = False) -> TrainedModel:
    with load_model_lock:
        path = os.path.join(MODEL_ZOO_PATH, id)
        if not os.path.exists(os.path.join(path, BEST_MODEL_FILE_NAME)):
            if id not in _get_all_model_ids():
                raise ValueError(f'Model with id {id} is not found on the server.')
            url_to_model_dir = f'{MODEL_DIR_URL}/{id}'
            if not os.path.exists(path):
                os.makedirs(path)
            for model_metadata_file in MODEL_METADATA_FILES:
                download_metadata(f'{url_to_model_dir}/{model_metadata_file}', path, model_metadata_file,
                                  version_to_convert_to=CONFIG_VERSION)
            if not load_description_only:
                logger.info(f'Model is not found in cache. Downloading from {url_to_model_dir} ...')
                for model_data_file in MODEL_DATA_FILES:
                    download_model(f'{url_to_model_dir}/{model_data_file}',
                                   os.path.join(path, model_data_file))

        return load_from_path(path, force_use_cpu, load_description_only)


def load_model_with_tag(tag: str, force_use_cpu: bool = False) -> TrainedModel:
    for cached in [True, False]:  # first checking tags of cached models
        for model in query_all_models(cached=cached):
            if model.is_tagged_by(tag):
                return load_model_by_id(model.id, force_use_cpu=force_use_cpu)
    raise ValueError(f'Model tagged with {tag} not found.')


def load_default_model(force_use_cpu: bool = False) -> TrainedModel:
    return load_model_with_tag('DEFAULT', force_use_cpu=force_use_cpu)