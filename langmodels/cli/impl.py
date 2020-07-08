from pathlib import Path
from typing import Dict, Optional, Any

import jsons

from langmodels.util.cuda import CudaNotAvailable, DeviceOptions
from langmodels.evaluation import evaluate_on_path
from langmodels.lmconfig.datamodel import LMTrainingConfig
from langmodels.lmconfig.patch import patch_config
from langmodels.lmconfig.serialization import load_config_or_metrics_from_file
from langmodels.repository import load_from_path
from langmodels.training.training import train


def get_option(args: Dict, option: str) -> Optional[Any]:
    return args[option] if option in args else None


def is_option_true(args: Dict, option: str) -> bool:
    return bool(get_option(args, option))


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
    rewrite_output = is_option_true(args, '--rewrite-output')
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
              allow_unks=allow_unks, output_path=output_path, rewrite_output=rewrite_output)
    except CudaNotAvailable:
        print('Gpu with CUDA-support is not available on this machine. '
              'Use --fallback-to-cpu  switch if you want to train on cpu')
        exit(4)


def handle_evaluation(args) -> None:
    after_epoch = get_option(args, '--after-epoch')
    device = get_option(args, '--device')
    device = int(device) if device else 0
    model = load_from_path(get_option(args, '<path-to-model>'), after_epoch=after_epoch, device=device)
    batch_size = get_option(args, '--batch-size')
    kw = {"batch_size": int(batch_size)} if batch_size else {}
    path = get_option(args, '--path')
    output_path = get_option(args, '--output-path')
    evaluation = evaluate_on_path(model, path, Path(output_path),
                     full_tokens=not is_option_true(args, '--sub-tokens'), **kw)
    print(evaluation.total())
