import logging
import logging.config
import os
import yaml

current_script_location = os.path.realpath(__file__)
root_package_dir = os.path.dirname(current_script_location)
project_dir = os.path.dirname(root_package_dir)


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


def _get_version():
    with open(os.path.join(root_package_dir, 'VERSION')) as version_file:
        return version_file.read().strip()


__version__ = _get_version()
