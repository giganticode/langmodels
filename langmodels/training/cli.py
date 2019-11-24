import importlib
import importlib.util
# comet.ml must be imported before everything else
module = importlib.import_module('comet_ml')

import jsons

from langmodels.lmconfig.patch import patch_config
from langmodels.lmconfig.serialization import load_config_from_file

from typing import Dict, Optional, Any

from langmodels import app_name

import docopt_subcommands as dsc

from langmodels.training.training import train
from langmodels.cuda_util import CudaNotAvailable
from langmodels.lmconfig.datamodel import DeviceOptions, LMTrainingConfig
from langmodels import __version__


def get_option(args: Dict, option: str) -> Optional[Any]:
    return args[option] if option in args else None


def is_option_true(args: Dict, option: str) -> bool:
    return bool(get_option(args, option))


@dsc.command()
def train_handler(args):
    """usage: {program} train [--config <config> | --patch <patch>] [--fallback-to-cpu] [--tune] [--disable-comet] [--device=<device>]

    Trains a language model according to the given config.

    Options:
      -C, --fallback-to-cpu                        Fallback to cpu if gpu with CUDA-support is not available
      -x, --disable-comet                          Do not log experiment to comet.ml
      -t, --tune                                   Training will be done only on a few batches
                                                    (can be used for model params such as batch size to make sure
                                                    the model fits into memory)
      -d <device>, --device=<device>               Device id to use
      -c, --config=<config>                        Path to the json with config to be used to train the model
      -p, --patch=<patch>                          'Patch' to apply to the default lm training config e.g

    """
    handle_train(args)


def parse_patch(patch_string: str) -> Dict[str, str]:
    return {l[0]: l[1] for l in [param.split(':') for param in patch_string.split(',')]}


def handle_train(args) -> None:
    fallback_to_cpu = is_option_true(args, '--fallback-to-cpu')
    tune = is_option_true(args, '--tune')
    comet = not is_option_true(args, '--disable-comet')
    device = get_option(args, '--device')
    device = int(device) if device else 0
    path_to_config = get_option(args, '--config')
    try:
        lm_training_config = load_config_from_file(path_to_config) if path_to_config else LMTrainingConfig()
    except jsons.exceptions.DecodeError:
        raise ValueError(f"Could not deserialize a valid config from {path_to_config}")

    patch = get_option(args, '--patch')
    lm_training_config = patch_config(lm_training_config, parse_patch(patch)) if patch else lm_training_config

    device_options = DeviceOptions(fallback_to_cpu=fallback_to_cpu, non_default_device_to_use=device)

    try:
        train(training_config=lm_training_config, device_options=device_options, tune=tune, comet=comet)
    except CudaNotAvailable:
        print('Gpu with CUDA-support is not available on this machine. '
              'Use --fallback-to-cpu  switch if you want to train on gpu')
        exit(4)


def run(args):
    dsc.main(app_name, f'{app_name} {__version__}', argv=args, exit_at_end=False)
