import hashlib
import os
from getpass import getpass
from threading import Lock
from typing import Optional, List, Tuple

import pysftp
import requests

from langmodels import MODEL_ZOO_PATH
from langmodels.repository.load import _get_all_model_ids, MODEL_DATA_FILES, MODEL_METADATA_FILES
from langmodels.repository.settings import MODEL_LIST_URL, PATH_TO_MODELS_ON_SERVER, SERVER_HOST_NAME, \
    PATH_TO_LOGS_ON_SERVER
from langmodels.util import HOME
from langmodels.model import TAGS_FILE_NAME


def upload_model_to_registry(id: str, tags: Optional[List[str]] = None,
                             username: Optional[str] = 'hbabii', password: Optional[str] = None) -> None:
    with remote_lock:
        local_path_to_model = os.path.join(MODEL_ZOO_PATH, id)
        if not os.path.exists(local_path_to_model):
            raise ValueError(f'Such model has not been trained: {id}')

        if id in _get_all_model_ids():
            raise ValueError(f'Model with such id already exists in the repository: {id}')

        tags = tags if tags else []
        with open(os.path.join(local_path_to_model, TAGS_FILE_NAME), 'w') as f:
            f.write(','.join(tags))
        md5_checksums: List[Tuple[str, str]] = []
        for file in MODEL_DATA_FILES:
            md5 = hashlib.md5(open(os.path.join(local_path_to_model, file), 'rb').read()).hexdigest()
            with open(os.path.join(local_path_to_model, file) + '.md5', 'w') as f:
                f.write(md5)
            md5_checksums.append((file, md5))

        _upload_files(local_path_to_model, PATH_TO_MODELS_ON_SERVER, model_id=id,
                      metadata_filenames=MODEL_METADATA_FILES, data_filenames_with_checksums=md5_checksums,
                      username=username, password=password)


def _upload_files(local_path: str, path_to_models_on_server: str, model_id: str,
                  metadata_filenames: List[str], data_filenames_with_checksums: List[Tuple[str, str]],
                  username: Optional[str] = None, password: Optional[str] = None) -> None:
    username = input('Username:') if not username else username
    password = getpass(f'Password for user {username}:') if not password else password

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
                uploaded_md5 = hashlib.md5(open(os.path.join(local_path, filename), 'rb').read()).hexdigest()
                if uploaded_md5 == checksum:
                    conn.put(os.path.join(local_path, filename) + '.md5')
                    conn.put(os.path.join(local_path, filename))
                else:
                    conn.remove(path_to_model)
                    return
        content = requests.get(MODEL_LIST_URL).content.decode(encoding='utf8')
        model_list = content.rstrip('\n').split('\n')
        model_list.append(model_id)
        path_to_local_list = os.path.join(local_path, os.path.basename(MODEL_LIST_URL))
        with open(path_to_local_list, 'w') as f:
            f.write('\n'.join(model_list))
        with conn.cd(path_to_models_on_server):
            conn.put(path_to_local_list)
        os.remove(path_to_local_list)
        #TODO proper handling and cleaning up in case of connection interruption etc.


remote_lock = Lock()
