import hashlib
import logging
import os
from getpass import getpass
from threading import Lock

import pysftp
import requests
from columnar import columnar
from typing import List, Optional, Mapping, Tuple

from langmodels import MODEL_ZOO_PATH
from langmodels.model import TrainedModel, ModelDescription
from langmodels.util import HOME

logger = logging.getLogger(__name__)

SERVER_URL = 'https://www.inf.unibz.it'
MODEL_DIR_URL = SERVER_URL + '/~hbabii/pretrained_models'
LIST_PREFIX = '.list'
MODEL_LIST_URL = MODEL_DIR_URL + LIST_PREFIX

SERVER_HOST_NAME = 'actarus.inf.unibz.it'
PATH_TO_MODELS_ON_SERVER = '/home/students/hbabii/public_html/pretrained_models'

PATH_TO_LOGS_ON_SERVER = '.pysftp.log'


lock = Lock()
remote_lock = Lock()


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


MODEL_DATA_FILES = ['best.pth', 'vocab']
MODEL_METADATA_FILES = ['config', 'metrics', 'tags']


def load_model_by_id(id: str, force_use_cpu: bool = False, load_description_only: bool = False) -> TrainedModel:
    with lock:
        path = os.path.join(MODEL_ZOO_PATH, id)
        if not os.path.exists(os.path.join(path, 'best.pth')):
            if id not in _get_all_model_ids():
                raise ValueError(f'Model with id {id} is not found on the server.')
            url_to_model_dir = MODEL_DIR_URL + '/' + id
            if not os.path.exists(path):
                os.makedirs(path)
            for model_metadata_file in MODEL_METADATA_FILES:
                _download_file(f'{url_to_model_dir}/{model_metadata_file}', os.path.join(path, model_metadata_file))
            if not load_description_only:
                logger.info(f'Model is not found in cache. Downloading from {url_to_model_dir} ...')
                for model_data_file in MODEL_DATA_FILES:
                    _download_file(f'{url_to_model_dir}/{model_data_file}',
                                   os.path.join(path, model_data_file), check_md5=True)

        return load_from_path(path, force_use_cpu, load_description_only)


def _upload_files(local_path: str, path_to_models_on_server: str, model_id: str,
                  metadata_filenames: List[str], data_filenames_with_checksums: Tuple[str, str],
                  username: Optional[str] = None, password: Optional[str] = None) -> None:
    username = input('Username:') if not username else username
    password = getpass('Password:') if not password else password

    cnopts = pysftp.CnOpts()
    cnopts.hostkeys.load(os.path.join(HOME, '.ssh', 'known_hosts'))

    with pysftp.Connection(host=SERVER_HOST_NAME, username=username, password=password,
                           cnopts=cnopts, log=PATH_TO_LOGS_ON_SERVER) as conn:
        path_to_model = f'{path_to_models_on_server}/{model_id}'
        conn.mkdir(path_to_model)
        with conn.cd(path_to_model):
            for filename in metadata_filenames:
                conn.put(os.path.join(local_path, filename))
            for filename, checksum in data_filenames_with_checksums:
                uploaded_md5 = hashlib.md5(open(os.path.join(local_path, filename)).read()).hexdigest()
                if uploaded_md5 == checksum:
                    conn.put(os.path.join(local_path, filename) + '.md5')
                else:
                    conn.remove(path_to_model)
                    return
        content = requests.get(MODEL_LIST_URL).content.decode(encoding='utf8')
        model_list = content.rstrip('\n').split('\n')
        model_list.append(model_id)
        path_to_local_list = os.path.join(local_path, 'model_list')
        with open(path_to_local_list, 'w') as f:
            f.write('\n'.join(model_list))
        with conn.cd(path_to_models_on_server):
            conn.put(path_to_local_list)
        os.remove(path_to_local_list)
        #TODO proper handling and cleaning up in case of connection interruption etc.


def upload_model_to_registry(id: str, tags: Optional[List[str]] = None,
                             username: Optional[str] = 'hbabii', password: Optional[str] = None) -> None:
    with remote_lock:
        local_path_to_model = os.path.join(MODEL_ZOO_PATH, id)
        if not os.path.exists(local_path_to_model):
            raise ValueError(f'Such model has not been trained: {id}')

        if id in _get_all_model_ids():
            raise ValueError(f'Model with such id already exists in the repository: {id}')

        path_on_server_to_model = f'{PATH_TO_MODELS_ON_SERVER}/{id}'
        tags = tags if tags else []
        with open(os.path.join(local_path_to_model, 'tags')) as f:
            f.write(','.join(tags))
        open(os.path.join(local_path_to_model, 'metrics')).close()
        md5_checksums = {}
        for file in MODEL_DATA_FILES:
            md5 = hashlib.md5(open(os.path.join(local_path_to_model, file)).read()).hexdigest()
            with open(os.path.join(local_path_to_model, file) + '.md5', 'w') as f:
                f.write(md5)
            md5_checksums[file] = md5

        _upload_files(os.path.join(local_path_to_model, MODEL_METADATA_FILES),
                      path_on_server_to_model, data_filenames_with_checksums=md5_checksums,
                      username=username, password=password)


def load_model_with_tag(tag: str, force_use_cpu: bool = False):
    for cached in [True, False]:  # first checking tags of cached models
        for model in query_all_models(cached=cached):
            if model.is_tagged_by(tag):
                return load_model_by_id(model.id, force_use_cpu=force_use_cpu)
    raise ValueError(f'Model tagged with {tag} not found.')


def load_default_model(force_use_cpu: bool = False) -> TrainedModel:
    return load_model_with_tag('DEFAULT', force_use_cpu=force_use_cpu)


__all__ = [load_default_model, list_pretrained_models, query_all_models, load_model_by_id, load_model_with_tag, load_from_path]
