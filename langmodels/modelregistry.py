import os
from langmodels.model import TrainedModel


MODEL_ZOO_PATH_ENV_VAR = 'MODEL_ZOO_PATH'

RANDOM_MODEL_NAME = 'dev_10k_1_10_190923.132328'
TINY_MODEL_NAME = 'langmodel-small-split_10k_1_512_190906.154943'
SMALL_MODEL_NAME = 'langmodel-medium-split_10k_1_512_190924.154542_-_langmodel-medium-split_10k_1_512_190925.141033'
SMALL_MODEL_1K = 'langmodel-medium-split_1k_1_512_190926.101149'
MEDIUM_MODEL_NAME = 'langmodel-large-split_10k_1_512_190926.120146'

if MODEL_ZOO_PATH_ENV_VAR in os.environ:
    MODEL_ZOO_PATH = os.environ[MODEL_ZOO_PATH_ENV_VAR]
else:
    MODEL_ZOO_PATH = os.path.join(os.environ['HOME'], 'modelzoo')


def from_path(path: str, force_use_cpu: bool = False) -> TrainedModel:
    return TrainedModel(path, force_use_cpu)


def get_default_model(force_use_cpu: bool = False) -> TrainedModel:
    return get_medium_model(force_use_cpu)


def get_random_model(force_use_cpu: bool = False) -> TrainedModel:
    return load_model_by_name(RANDOM_MODEL_NAME, force_use_cpu)


def get_tiny_model(force_use_cpu: bool = False) -> TrainedModel:
    return load_model_by_name(TINY_MODEL_NAME, force_use_cpu)


def get_small_model(force_use_cpu: bool = False) -> TrainedModel:
    return load_model_by_name(SMALL_MODEL_NAME, force_use_cpu)


def get_medium_model(force_use_cpu: bool = False) -> TrainedModel:
    return load_model_by_name(MEDIUM_MODEL_NAME, force_use_cpu)


def load_model_by_name(name: str, force_use_cpu: bool = False) -> TrainedModel:
    path = os.path.join(MODEL_ZOO_PATH, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path does not exist: {path}. '
                                f'Did you set {MODEL_ZOO_PATH_ENV_VAR} env var correctly? '
                                f'Your current {MODEL_ZOO_PATH_ENV_VAR} path is {MODEL_ZOO_PATH}')
    return from_path(path, force_use_cpu)
