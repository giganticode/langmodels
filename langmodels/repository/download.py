import hashlib
import logging
import os

import requests

from langmodels.repository.convert import convert_file_version

logger = logging.getLogger(__name__)


def _download_file(url: str, path: str) -> None:
    response = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)


def download_metadata(url: str, path: str, file_name: str, version_to_convert_to: str) -> None:
    path = os.path.join(path, file_name)
    tmp_path = path + '.tmp'
    _download_file(url, tmp_path)

    convert_file_version(tmp_path, file_name, version_to_convert_to)

    os.replace(tmp_path, path)


def download_model(url: str, path: str) -> None:
    _download_file(url, path + '.tmp')

    correct_md5 = requests.get(url + ".md5").content.decode(encoding='utf8').rstrip('\n')
    md5 = hashlib.md5(open(path + '.tmp', 'rb').read()).hexdigest()
    if md5 != correct_md5:
        logger.warning(f'The md5 checksum for {url} is not correct: {md5}. Downloading the file again...')
        download_model(url, path)

    os.replace(path + '.tmp', path)