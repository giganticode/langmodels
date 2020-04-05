import logging
import os
from pprint import pformat
from dataclasses import dataclass
from typing import Optional

import jsons
from comet_ml import Experiment
from fastai.text import Vocab
from flatdict import FlatDict

from langmodels import MODEL_ZOO_PATH
from langmodels.cuda_util import DeviceOptions
from langmodels.file_util import check_path_exists, check_path_writable
from langmodels.lmconfig.datamodel import LMTrainingConfig, BEST_MODEL_FILE_NAME, create_comet_experiment, \
    TransformerArch, TMP_SUFFIX
from langmodels.lmconfig.serialization import dump_to_file
from langmodels.model import VOCAB_FILE_NAME, CONFIG_FILE_NAME

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun(object):
    id: str
    config: LMTrainingConfig
    gpu: DeviceOptions
    comet_experiment: Optional[Experiment]
    first_model_trained: bool = False

    def __post_init__(self):
        if self.config.base_model:
            check_path_exists(os.path.join(self.config.base_model, BEST_MODEL_FILE_NAME))
        check_path_writable(self.path_to_trained_model)

    @classmethod
    def with_config(cls, config: LMTrainingConfig, device_options: DeviceOptions = DeviceOptions(), comet: bool = True):
        run_id = ExperimentRun._generate_run_id(config)
        comet_experiment = create_comet_experiment(run_id) if comet else None
        return cls(run_id, config, device_options, comet_experiment)

    def set_first_model_trained(self):
        self.first_model_trained = True

    @staticmethod
    def _generate_run_id(config: LMTrainingConfig) -> str:
        name_parts = []
        if config.base_model:
            name_parts.append([os.path.basename(config.base_model)])

        dataset = os.path.basename(config.corpus.path)
        if config.prep_function.params:
            prep_func_param = config.prep_function.params[0]
        else:
            prep_func_param = config.prep_function.callable.__name__
        n_layers = config.arch.n_layers
        n_hid = config.arch.n_hid if not isinstance(config.arch, TransformerArch) \
            else config.arch.d_inner

        import datetime
        time_now = datetime.datetime.now()
        timestamp = f"{time_now:%y%m%d.%H%M%S}"

        name_parts.append([dataset, str(prep_func_param), str(n_layers), str(n_hid), timestamp])

        return "_-_".join(map(lambda p: "_".join(p), name_parts))

    @property
    def path_to_trained_model(self) -> str:
        return self.perm_path_to_model if self.first_model_trained else f'{self.perm_path_to_model}{TMP_SUFFIX}'

    @property
    def perm_path_to_model(self) -> str:
        return os.path.join(MODEL_ZOO_PATH, self.id)

    def log_vocab(self, vocab: Vocab) -> None:
        logger.info(f"Vocab size: {len(vocab.itos)}")
        vocab.save(os.path.join(self.path_to_trained_model, VOCAB_FILE_NAME))
        if self.comet_experiment:
            self.comet_experiment.log_parameter("vocabulary", len(vocab.itos))

    def log_experiment_input(self) -> None:
        logger.info(f'Using the following config: \n{pformat(jsons.dump(self.config))}')
        dump_to_file(self.config, os.path.join(self.path_to_trained_model, CONFIG_FILE_NAME))
        if self.comet_experiment:
            flat_config = FlatDict(jsons.dump(self.config))
            for name, value in flat_config.items():
                self.comet_experiment.log_parameter(name, value)
            self.comet_experiment.log_metric(MODEL_AVAILABLE_METRIC_NAME, False)
            self.comet_experiment.log_metric(TERMINATED_NORMALLY_METRIC_NAME, False)


MODEL_AVAILABLE_METRIC_NAME = 'model available'
TERMINATED_NORMALLY_METRIC_NAME = 'terminated normally'