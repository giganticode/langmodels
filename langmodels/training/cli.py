import importlib
# comet.ml must be imported before everything else
from langmodels.cuda_util import CudaNotAvailable

module = importlib.import_module('comet_ml')

from typing import Dict, Optional, Any

from langmodels import app_name

import docopt_subcommands as dsc
from langmodels.training.training import train

from langmodels.lmconfig.datamodel import DeviceOptions, LMTrainingConfig
from langmodels import __version__


def get_option(args: Dict, option: str) -> Optional[Any]:
    return args[option] if option in args else None


def is_option_true(args: Dict, option: str) -> bool:
    return bool(get_option(args, option))


@dsc.command()
def train_handler(args):
    """usage: {program} train [--config=<config>] [--fallback-to-cpu] [--tune] [--disable-comet] [--device=<device>]

    Trains a language model according to the given config.

    Options:
      -p, --fallback-to-cpu                        Fallback to cpu if gpu with CUDA-support is not available
      -x, --disable-comet                          Do not log experiment to comet.ml
      -t, --tune                                   Training will be done only on a few batches
                                                    (can be used for model params such as batch size to make sure
                                                    the model fits into memory)
      -d <device>, --device=<device>               Device id to use
      --config=<config>               Name of the config to use to train the model
    """
    handle_train(args)


CONFIG_VAR_NAME = 'lm_training_config'


def load_training_config(config_name: str) -> LMTrainingConfig:
    try:
        module = importlib.import_module(f'langmodels.lmconfig.{config_name}')
    except ModuleNotFoundError:
        print(f'Config {config_name} not found')
        exit(2)
    try:
        return getattr(module, CONFIG_VAR_NAME)
    except AttributeError:
        print(f'Config module must contain {CONFIG_VAR_NAME} variable')
        exit(3)


def handle_train(args) -> None:
    fallback_to_cpu = is_option_true(args, '--fallback-to-cpu')
    tune = is_option_true(args, '--tune')
    comet = not is_option_true(args, '--disable-comet')
    device = get_option(args, '--device')
    device = int(device) if device else 0
    config = get_option(args, '--config')
    lm_training_config = load_training_config(config) if config else LMTrainingConfig()

    device_options = DeviceOptions(fallback_to_cpu=fallback_to_cpu, non_default_device_to_use=device)

    try:
        train(training_config=lm_training_config, device_options=device_options, tune=tune, comet=comet)
    except CudaNotAvailable:
        print('Gpu with CUDA-support is not available on this machine. '
              'Use --fallback-to-cpu  switch if you want to train on gpu')
        exit(4)


def run(args):
    dsc.main(app_name, f'{app_name} {__version__}', argv=args, exit_at_end=False)
