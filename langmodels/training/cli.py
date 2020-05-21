import jsons

from langmodels.lmconfig.patch import patch_config
from langmodels.lmconfig.serialization import load_config_or_metrics_from_file

from typing import Dict, Optional, Any

from langmodels import app_name

import docopt_subcommands as dsc

from langmodels.training.training import train
from langmodels.cuda_util import CudaNotAvailable, DeviceOptions
from langmodels.lmconfig.datamodel import LMTrainingConfig
from langmodels import __version__


def get_option(args: Dict, option: str) -> Optional[Any]:
    return args[option] if option in args else None


def is_option_true(args: Dict, option: str) -> bool:
    return bool(get_option(args, option))


@dsc.command()
def train_handler(args):
    """usage: {program} train [--config <config>] [--patch <patch>] [--fallback-to-cpu] [--tune] [--disable-comet]
    [--save-every-epoch] [--allow-unks] [--device=<device>] [--output-path <path>]

    Trains a language model according to the given config.

    Options:
      -C, --fallback-to-cpu                        Fallback to cpu if gpu with CUDA-support is not available
      -x, --disable-comet                          Do not log experiment to comet.ml
      -e, --save-every-epoch                       Save the model to the disk after every epoch
      -u, --allow_unks                             Allow unknown tokens
      -t, --tune                                   Training will be done only on a few batches
                                                    (can be used for model params such as batch size to make sure
                                                    the model fits into memory)
      -d <device>, --device=<device>               Device id to use
      -c, --config=<config>                        Path to the json with config to be used to train the model
      -p, --patch=<patch>                          'Patch' to apply to the default lm training config e.g
      -o, --output-path=<path>                     Path to where the models and metrics will be saved.
                                                   If not specified:
                                                   On Mac OS X:
                                                       ~/Library/Application Support/langmodels/<langmodels-version>/modelzoo
                                                   On Unix:
                                                       ~/.local/share/langmodels/<langmodels-version>/modelzoo
                                                       or if XDG_DATA_HOME is defined:
                                                       $XDG_DATA_HOME/langmodels/<langmodels-version>/modelzoo

    """
    handle_train(args)


def parse_patch(patch_string: str) -> Dict[str, str]:
    return {l[0]: l[1] for l in [param.split('=') for param in patch_string.split(',')]}


def handle_train(args) -> None:
    fallback_to_cpu = is_option_true(args, '--fallback-to-cpu')
    tune = is_option_true(args, '--tune')
    comet = not is_option_true(args, '--disable-comet')
    save_every_epoch = is_option_true(args, '--save-every-epoch')
    allow_unks = is_option_true(args, '--allow-unks')
    device = get_option(args, '--device')
    device = int(device) if device else 0
    path_to_config = get_option(args, '--config')
    output_path = get_option(args, '--output-path')
    try:
        lm_training_config = load_config_or_metrics_from_file(path_to_config, LMTrainingConfig) if path_to_config else LMTrainingConfig()
    except jsons.exceptions.DecodeError:
        raise ValueError(f"Could not deserialize a valid config from {path_to_config}")

    patch = get_option(args, '--patch')
    lm_training_config = patch_config(lm_training_config, parse_patch(patch)) if patch else lm_training_config

    device_options = DeviceOptions(fallback_to_cpu=fallback_to_cpu, non_default_device_to_use=device)

    try:
        train(training_config=lm_training_config, device_options=device_options,
              tune=tune, comet=comet, save_every_epoch=save_every_epoch,
              allow_unks=allow_unks, output_path=output_path)
    except CudaNotAvailable:
        print('Gpu with CUDA-support is not available on this machine. '
              'Use --fallback-to-cpu  switch if you want to train on gpu')
        exit(4)


def run(args):
    dsc.main(app_name, f'{app_name} {__version__}', argv=args, exit_at_end=False)
