import os
from typing import Dict

import jsons
from jq import jq
from langmodels import project_dir

from langmodels.repository.convert import convert_dict

config_v002_gru_cosine = {"arch": {
    "bidir": False, "clip": 0.3,
    "adam_betas": [0.7, 0.99],
    "reg_fn": {"alpha": 2.1, "beta": 1.1},
    "drop": {"multiplier": 0.5, "out": 0.1, "oute": 0.02, "outh": 0.15, "outi": 0.25, "w": 0.2},
    "emb_sz": 10, "n_hid": 10, "n_layers": 1, "out_bias": True, "tie_weights": True},

    "base_model": None, "bptt": 10, "bs": 5, "config_version": "0.0.2-alpha.0",
    "corpus": {"extensions": "java", "path": "/home/lv71161/hlibbabii/raw_datasets/dev"},
    "prep_function": {"callable": "bpe", "params": ["10k"],
                      "options": {"no_str": False, "no_com": False, "no_spaces": True, "no_unicode": True, "max_str_length": 922337203}},
    "training_procedure": {
        "schedule": {"cyc_len": 3, "early_stop": {"patience": 3}, "max_epochs": 1, "max_lr": 0.0001},
        "weight_decay": 1e-06}}

config_v002_lstm_rafael = {"arch": {
    "bidir": False, "clip": 0.3,
    "adam_betas": [0.9, 0.99],
    "reg_fn": {"alpha": 2.1, "beta": 1.1}, "qrnn": False,
    "drop": {"multiplier": 0.5, "out": 0.1, "oute": 0.02, "outh": 0.15, "outi": 0.25, "w": 0.2},
    "emb_sz": 10, "n_hid": 10, "n_layers": 1, "out_bias": True, "tie_weights": True},

    "base_model": None, "bptt": 10, "bs": 5, "config_version": "0.0.2-alpha.0",
    "corpus": {"extensions": "java", "path": "/home/lv71161/hlibbabii/raw_datasets/dev"},
    "prep_function": {"callable": "bpe", "params": ["10k"],
                      "options": {"no_str": False, "no_com": False, "no_spaces": True, "no_unicode": True, "max_str_length": 922337203}},
    "training_procedure": {
        "schedule": {"mult_coeff": 0.5, "max_epochs": 1, "init_lr": 0.0001, "patience": 3, "max_lr_reduction_times": 6},
        "weight_decay": 1e-06}}


config_v003_gru_cosine = {"arch": {
    "name": "gru",
    "bidir": False,
    "drop": {"multiplier": 0.5, "out": 0.1, "oute": 0.02, "outh": 0.15, "outi": 0.25, "w": 0.2},
    "emb_sz": 10, "n_hid": 10, "n_layers": 1, "out_bias": True, "tie_weights": True},
"base_model": None, "bptt": 10, "bs": 5, "config_version": "0.0.3-alpha.0",
"corpus": {"extensions": "java", "path": "/home/lv71161/hlibbabii/raw_datasets/dev"},
"prep_function": {"callable": "bpe", "params": ["10k"],
    "options": {"no_str": False, "no_com": False, "no_spaces": True, "no_unicode": True, "max_str_length": 922337203}},
"training": {
    "gradient_clip": 0.3,
    "activation_regularization": {"alpha": 2.1, "beta": 1.1},
    "optimizer": {"name": "Adam", "betas": [0.7, 0.99]},
    "schedule": {"name": "cosine", "cyc_len": 3, "early_stop": {"patience": 3}, "max_epochs": 1, "max_lr": 0.0001},
    "files_per_epoch": 50000,
    "weight_decay": 1e-06}}

config_v003_lstm_rafael = {"arch": {
    "name": "lstm",
    "bidir": False, "qrnn": False,
    "drop": {"multiplier": 0.5, "out": 0.1, "oute": 0.02, "outh": 0.15, "outi": 0.25, "w": 0.2},
    "emb_sz": 10, "n_hid": 10, "n_layers": 1, "out_bias": True, "tie_weights": True},
    "base_model": None, "bptt": 10, "bs": 5, "config_version": "0.0.3-alpha.0",
    "corpus": {"extensions": "java", "path": "/home/lv71161/hlibbabii/raw_datasets/dev"},
    "prep_function": {"callable": "bpe", "params": ["10k"],
                      "options": {"no_str": False, "no_com": False, "no_spaces": True, "no_unicode": True, "max_str_length": 922337203}},
    "training": {
        "gradient_clip": 0.3,
        "activation_regularization": {"alpha": 2.1, "beta": 1.1},
        "optimizer": {"name": "Adam", "betas": [0.9, 0.99]},
        "schedule": {"name": "rafael", "mult_coeff": 0.5, "max_epochs": 1, "init_lr": 0.0001, "patience": 3, "max_lr_reduction_times": 6},
        "files_per_epoch": 50000,
        "weight_decay": 1e-06}}

metrics_v002 = {"bin_entropy": 2.1455788479, "n_epochs": 6, "best_epoch": 5, "training_time_minutes_per_epoch": 1429, "trainable_params": 27726250, "size_on_disk_mb": 350, "config_version": "0.0.2-alpha.0"}


metrics_v003 = {"bin_entropy": 2.1455788479, "n_epochs": 6, "best_epoch": 5, "training_time_minutes_per_epoch": 1429, "trainable_params": 27726250, "size_on_disk_mb": 350, "config_version": "0.0.3-alpha.0"}


def test_003_to_002():
    assert convert_dict(config_v003_gru_cosine, 'config', '0.0.2-alpha.0') == config_v002_gru_cosine
    assert convert_dict(config_v003_lstm_rafael, 'config', '0.0.2-alpha.0') == config_v002_lstm_rafael
    assert convert_dict(metrics_v003, 'metrics', '0.0.2-alpha.0') == metrics_v002


def _get_transformation_dict(version: str) -> Dict[str, str]:
    path_to_tranformation_string = os.path.join(project_dir, 'converters', 'forward', f'{version}.jq')
    with open(path_to_tranformation_string, 'r') as f:
        serialized_transformation_dict = f.read()
    transformation_dict = jsons.loads(serialized_transformation_dict)
    return transformation_dict


def test_002_to_003():
    version = '0.0.2-alpha.0'
    tranformation_string = _get_transformation_dict(version)

    assert jq(tranformation_string['config']).transform(config_v002_gru_cosine) == config_v003_gru_cosine
    assert jq(tranformation_string['config']).transform(config_v002_lstm_rafael) == config_v003_lstm_rafael
    assert jq(tranformation_string['metrics']).transform(metrics_v002) == metrics_v003
