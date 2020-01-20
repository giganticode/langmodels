import os
import re

import torch
from functools import partial

import jsons
import sys
from dataclasses import dataclass, field, asdict

from comet_ml import Experiment
from torch import optim
from typing import Optional, Callable, Tuple, List, Type, Dict, Any

import dataprep.api.corpus as corpus_api
from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import AWD_LSTM, Transformer, Activation

from langmodels import MODEL_ZOO_PATH, __version__, __major_version__
from langmodels.nn import GRU
from langmodels.util import HOME

CONFIG_VERSION = __major_version__ if __major_version__ > 0 else __version__

TMP_SUFFIX = '.tmp'


@dataclass(frozen=True)
class Dropouts(object):
    multiplier: float = 0.5
    oute: float = 0.02
    outi: float = 0.25
    outh: float = 0.15
    w: float = 0.2
    out: float = 0.1


@dataclass(frozen=True)
class ActivationRegularization(object):
    alpha: float = 2.
    beta: float = 1.


@dataclass(frozen=True)
class TrainingSchedule(object):
    name: str
    _serializer: Any = None

    @classmethod
    def get_serializer(cls):
        if not cls._serializer:
            cls._serializer = jsons.fork()

        return cls._serializer


@dataclass(frozen=True)
class RafaelsTrainingSchedule(TrainingSchedule):
    name: str = 'rafael'
    init_lr: float = 1e-4
    mult_coeff: float = 0.5
    max_epochs: int = 50
    max_lr_reduction_times: int = 6
    patience: int = 0


@dataclass(frozen=True)
class EarlyStop(object):
    patience: int = 3


@dataclass(frozen=True)
class CosineLRSchedule(TrainingSchedule):
    name: str = 'cosine'
    max_lr: float = 1e-4
    cyc_len: int = 3
    max_epochs: int = 30
    early_stop: EarlyStop = EarlyStop()


@dataclass(frozen=True)
class Corpus(object):
    path: str = os.path.join(HOME, 'dataset')
    extensions: str = 'java'  # in format "py" or "java|c|py"


PrepCallable = Callable[..., PreprocessedCorpus]
ParametrizedPrepCallable = Callable[[Corpus], PreprocessedCorpus]


@dataclass(frozen=True)
class PrepFunctionOptions(object):
    no_unicode: bool = True
    no_spaces: bool = True
    no_com: bool = False
    no_str: bool = False
    max_str_length: int = sys.maxsize


@dataclass(frozen=True)
class PrepFunction(object):
    callable: PrepCallable = corpus_api.bpe
    params: List[str] = field(default_factory=lambda: ['10k'])
    options: PrepFunctionOptions = PrepFunctionOptions()

    @property
    def apply(self) -> ParametrizedPrepCallable:
        def prep_corpus(corpus: Corpus, **kwargs) -> PreprocessedCorpus:
            return self.callable(corpus.path, *self.params, **asdict(self.options), **kwargs,
                                 extensions=corpus.extensions)

        return prep_corpus

    @staticmethod
    def serializer(prep_function: 'PrepFunction', **kwargs) -> Dict[str, Any]:
        return {'callable': prep_function.callable.__name__,
                'params': prep_function.params,
                'options':  jsons.dump(prep_function.options)}

    @staticmethod
    def deserializer(dct: Dict[str, Any], cls: Type['PrepFunction'], **kwargs) -> 'PrepFunction':
        import dataprep.api.corpus as api
        return cls(
            callable=getattr(api, dct['callable']),
            params=dct['params'],
            options=jsons.load(dct['options'], PrepFunctionOptions)
        )


@dataclass(frozen=True)
class Optimizer(object):
    name: str
    _serializer: Any = None

    @classmethod
    def get_serializer(cls):
        if not cls._serializer:
            cls._serializer = jsons.fork()

        return cls._serializer


@dataclass(frozen=True)
class SGD(Optimizer):
    name: str = 'sgd'
    momentum: float = 0.9

    def get_callable(self):
        return partial(torch.optim.SGD, momentum=self.momentum)


@dataclass(frozen=True)
class Adam(Optimizer):
    name: str = 'Adam'
    betas: Tuple[float, float] = (0.9, 0.99)

    def get_callable(self):
        return partial(optim.Adam, betas=self.betas)


def camel_case_to_snake_case(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def field_based_deserializer_func(dct: Dict[str, Any], cls: Type, **kwargs) -> Any:
    all_arch_classes = cls.__subclasses__()
    field_name = 'name'
    for arch_class in all_arch_classes:
        if getattr(arch_class, field_name) == dct[field_name]:
            try:
                return jsons.load(dct, arch_class, fork_inst=arch_class.get_serializer())
            except jsons.exceptions.DeserializationError as e:
                raise ValueError(f'Failed to deserialize: {dct}') from e
    raise ValueError(f'Unknown {cls.__name__}: {dct[field_name]}')


@dataclass(frozen=True)
class Arch(object):
    name: str
    bidir: bool = False
    emb_sz: int = 1024
    n_hid: int = 1024
    n_layers: int = 3
    drop: Dropouts = Dropouts()
    tie_weights: bool = True
    out_bias: bool = True

    _serializer: Any = None

    @classmethod
    def get_serializer(cls):
        if not cls._serializer:
            cls._serializer = jsons.fork()

        return cls._serializer


@dataclass(frozen=True)
class LstmArch(Arch):
    name: str = 'lstm'
    qrnn: bool = False

    def get_module(self):
        return AWD_LSTM


@dataclass(frozen=True)
class GruArch(Arch):
    name: str = 'gru'

    def get_module(self):
        return GRU


@dataclass(frozen=True)
class TransformerDropouts(object):
    multiplier: float = 1.0
    resid: float = 0.1
    attn: float = 0.1
    ff: float = 0.1
    embed: float = 0.1
    output: float = 0.


@dataclass(frozen=True)
class TransformerArch(Arch):
    arch: str = 'transformer'
    ctx_len: int = 256
    n_layers: int = 3
    n_heads: int = 6
    d_model: int = 512
    d_head: int = 16
    d_inner: int = 2048
    drop: TransformerDropouts = TransformerDropouts()
    bias: bool = True
    scale: bool = True
    act: Activation = Activation.GeLU
    double_drop: bool = False
    tie_weights: bool = True
    out_bias: bool = False
    mask: bool = True

    def get_module(self):
        return Transformer


@dataclass(frozen=True)
class Training(object):
    optimizer: Optimizer = Adam()
    weight_decay: float = 1e-6
    gradient_clip: float = 0.3
    activation_regularization: ActivationRegularization = ActivationRegularization()
    schedule: TrainingSchedule = RafaelsTrainingSchedule()
    files_per_epoch: Optional[int] = 50 * 1000


@dataclass(frozen=True)
class DeviceOptions(object):
    fallback_to_cpu: bool = False
    non_default_device_to_use: Optional[int] = None


@dataclass(frozen=True)
class LMTrainingConfig(object):
    corpus: Corpus = Corpus()
    base_model: Optional[str] = None
    bs: int = 32
    prep_function: PrepFunction = PrepFunction()
    arch: Arch = LstmArch()
    bptt: int = 200
    training: Training = Training()
    config_version: str = CONFIG_VERSION
    _serializer: Any = None

    @classmethod
    def get_serializer(cls):
        if not LMTrainingConfig._serializer:
            cls._serializer = jsons.fork()
            jsons.set_deserializer(jsons.default_object_deserializer, cls=cls, fork_inst=cls._serializer)
            jsons.set_deserializer(field_based_deserializer_func, cls=Arch, fork_inst=cls._serializer)
            jsons.set_deserializer(field_based_deserializer_func, cls=TrainingSchedule, fork_inst=cls._serializer)
            jsons.set_deserializer(field_based_deserializer_func, cls=Optimizer, fork_inst=cls._serializer)
            jsons.set_deserializer(PrepFunction.deserializer, cls=PrepFunction, fork_inst=cls._serializer)

            jsons.set_serializer(jsons.default_object_serializer, cls=cls, fork_inst=cls._serializer)
            jsons.set_serializer(PrepFunction.serializer, cls=PrepFunction, fork_inst=cls._serializer)

        return cls._serializer

    @staticmethod
    def deserializer(dct: Dict[str, Any], cls: Type['LMTrainingConfig'], **kwargs) -> 'LMTrainingConfig':
        return jsons.load(dct, cls, fork_inst=cls.get_serializer())

    @staticmethod
    def serializer(config: 'LMTrainingConfig', **kwargs) -> Dict[str, Any]:
        return jsons.dump(config, fork_inst=LMTrainingConfig.get_serializer(), strip_privates=True)

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION:
            raise TypeError(f'Trying to deserealize '
                            f'{type(self).__name__} {self.config_version} '
                            f'to {type(self).__name__} {CONFIG_VERSION} object')


def create_comet_experiment(run_id: str):
    experiment = Experiment('8KN4SL1OwKieQYocdsDFX5Oi5')
    experiment.set_name(run_id)
    return experiment


@dataclass
class LMTrainingMetrics(object):
    bin_entropy: Optional[float] = None
    training_time_minutes_per_epoch: Optional[int] = None
    n_epochs: Optional[int] = None
    best_epoch: Optional[int] = None
    trainable_params: Optional[int] = None
    size_on_disk_mb: Optional[int] = None
    config_version: str = CONFIG_VERSION

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION:
            raise TypeError(f'Trying to deserealize '
                            f'{type(self).__name__} {self.config_version} '
                            f'to {type(self).__name__} {CONFIG_VERSION} object')


class ExperimentRun:
    def __init__(self, run_id: str, config: LMTrainingConfig,
                 device_options: DeviceOptions, comet_experiment: Optional):
        self.id = run_id
        self.config = config
        self.gpu = device_options
        self.comet_experiment: Optional[Experiment] = comet_experiment
        self.first_model_trained = False

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
        prep_func_param = config.prep_function.params[0]
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


jsons.set_serializer(LMTrainingConfig.serializer, cls=LMTrainingConfig)
jsons.set_deserializer(LMTrainingConfig.deserializer, cls=LMTrainingConfig)