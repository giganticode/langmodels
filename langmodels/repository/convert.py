import logging
from typing import Dict, Union, Any

import jsons
import requests
import semver

from langmodels import app_name
from langmodels.lmconfig.datamodel import LMTrainingMetrics, LMTrainingConfig
from langmodels.lmconfig.serialization import dump_to_file, load_config_or_metrics_form_dict
from langmodels.repository.settings import CONVERTERS_URL
from langmodels.model import CONFIG_FILE_NAME, METRICS_FILE_NAME

logger = logging.getLogger(__name__)

CONVERTABLE_METADATA_FILES = [CONFIG_FILE_NAME, METRICS_FILE_NAME]


def fetch_convertion_json(version: str) -> Dict[str, str]:
    json_url = f'{CONVERTERS_URL}/{version}.jq'
    response = requests.get(json_url)
    if response.status_code == 200:
        s = response.text.replace('\n', ' ').strip()
    else:
        raise ValueError(f'Cannot get version converter from {json_url}')
    return jsons.loads(s)


def convert_file_version(path: str, file_name: str, version_to_convert_to: str) -> None:
    if file_name not in CONVERTABLE_METADATA_FILES:
        return

    try:
        with open(path, 'r') as f:
            s = f.read().replace('\n', '')
            loaded_dict = jsons.loads(s)
    except jsons.exceptions.DecodeError:
        raise ValueError(f'Error when trying to load file: {path}.\n File contents:\n {s}')

    try:
        converted_dict: Dict[str, Any] = convert_dict(loaded_dict, file_name, version_to_convert_to)
    except VersionError as ve:
        raise VersionError(f"The version of the downloaded models "
                           f"cannot be lower than the version of the library ({version_to_convert_to}). "
                           f"Please contact the development team if you see this error.") from ve
    try:
        tp = LMTrainingMetrics if file_name == 'metrics' else LMTrainingConfig
        converted_config: Union[LMTrainingMetrics, LMTrainingConfig] = load_config_or_metrics_form_dict(converted_dict, tp)
    except jsons.exceptions.DeserializationError as ex:
        raise ValueError(f"Could not deserialize: {converted_dict}\n Path {path}") from ex

    dump_to_file(converted_config, path)


class VersionError(Exception):
    pass


def convert_dict(dct: Dict[str, Any], file_name: str, version_to_convert_to: str) -> Dict[str, Any]:
    try:
        version_in_file = dct['config_version']
        if semver.compare(version_in_file, version_to_convert_to) == 0:
            return dct
    except KeyError:
        raise ValueError(f"Cannot get config version from json:\n {dct}, 'config_version' attribute is not found.")

    try:
        from jq import jq
    except ImportError:
        logger.error(f"Downloaded models are the newer version. To convert them to the version of {app_name} "
                     f"you are using, jq needs to be installed. Please install jq (https://pypi.org/project/jq/) "
                     f"or use the latest version of {app_name}.")
        exit(47)
    if semver.compare(version_in_file, version_to_convert_to) < 0:
        raise VersionError(f"Something went wrong. "
                         f"The version of the downloaded models ({version_in_file}) "
                         f"cannot be lower than the version of the library ({version_to_convert_to}).")

    while semver.compare(version_in_file, version_to_convert_to) > 0:
        convertion_json = fetch_convertion_json(version_in_file)
        try:
            jq_transformation_string = convertion_json[file_name]
        except AttributeError:
            raise ValueError(f'Converter from version {version_in_file} doesn\'t know '
                             f'how to convert this file: {file_name}')
        loaded_dict = jq(jq_transformation_string).transform(dct)
        version_in_file = loaded_dict['config_version']
    return loaded_dict
