import logging
import logging.config
import os

import appdirs
import yaml
import matplotlib
matplotlib.use('PS')  # to avoid error on OSX - this should go here, before other imports

from langmodels.nn import add_gru_to_model_data

app_name = 'langmodels'

current_script_location = os.path.realpath(__file__)
root_package_dir = os.path.dirname(current_script_location)
project_dir = os.path.dirname(root_package_dir)


def _get_version():
    with open(os.path.join(root_package_dir, 'VERSION')) as version_file:
        return version_file.read().strip()


def _get_major_version():
    return int(__version__.split('.')[0])


__version__ = _get_version()
__major_version__ = _get_major_version()

user_data_dir = appdirs.user_data_dir(app_name, appauthor=False, version=_get_version())
MODEL_ZOO_PATH = os.path.join(user_data_dir, 'modelzoo')


def load_logging_config():
    path = os.path.join(root_package_dir, 'logging.yaml')
    if os.path.exists(path):
        with open(path, 'rt') as f:
            logging_config = yaml.safe_load(f.read())
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=logging.DEBUG)

load_logging_config()

logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('everett').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

add_gru_to_model_data()
