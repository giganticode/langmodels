import os
import sys
from dataclasses import dataclass, field, asdict

from comet_ml import Experiment
from typing import Optional, Callable, Tuple, Union, List, Type

import dataprep.api.corpus as corpus_api
from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import AWD_LSTM, Transformer, TransformerXL, Activation

from langmodels import MODEL_ZOO_PATH, __version__, __major_version__
from langmodels.nn import GRU

CONFIG_VERSION = __major_version__ if __major_version__ > 0 else __version__

HOME = os.environ['HOME']

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
class RegFn(object):
    alpha: float = 2.
    beta: float = 1.


@dataclass(frozen=True)
class TrainingSchedule(object):
    pass


@dataclass(frozen=True)
class RafaelsTrainingSchedule(TrainingSchedule):
    init_lr: float = 1e-4
    mult_coeff: float = 0.5
    max_epochs: int = 50
    max_lr_reduction_times: int = 6


@dataclass(frozen=True)
class EarlyStop(object):
    patience: int = 3


@dataclass(frozen=True)
class CosineLRSchedule(TrainingSchedule):
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


@dataclass(frozen=True)
class LstmArch(object):
    bidir: bool = False
    qrnn: bool = False
    emb_sz: int = 1024
    n_hid: int = 1024
    n_layers: int = 3
    adam_betas: Tuple[float, float] = (0.7, 0.99)
    clip: float = 0.3
    reg_fn: RegFn = RegFn()
    drop: Dropouts = Dropouts()
    tie_weights: bool = True
    out_bias: bool = True
    lstm: bool = True

    def __post_init__(self):
        if not self.lstm:
            raise TypeError(f'Value of lstm field in LstmArch must be set to True')


@dataclass(frozen=True)
class GruArch(object):
    bidir: bool = False
    emb_sz: int = 1024
    n_hid: int = 1024
    n_layers: int = 3
    adam_betas: Tuple[float, float] = (0.7, 0.99)
    clip: float = 0.3
    reg_fn: RegFn = RegFn()
    drop: Dropouts = Dropouts()
    tie_weights: bool = True
    out_bias: bool = True
    gru: bool = True

    def __post_init__(self):
        if not self.gru:
            raise TypeError(f'Value of gru field in GruArch must be set to True')


@dataclass(frozen=True)
class TransformerDropouts(object):
    multiplier: float = 1.0
    resid: float = 0.1
    attn: float = 0.1
    ff: float = 0.1
    embed: float = 0.1
    output: float = 0.


@dataclass(frozen=True)
class TransformerArch(object):
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


@dataclass(frozen=True)
class TrainingProcedure(object):
    schedule: Union[CosineLRSchedule, RafaelsTrainingSchedule] = RafaelsTrainingSchedule()
    files_per_epoch: Optional[int] = 50 * 1000
    weight_decay: float = 1e-6


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
    arch: Union[LstmArch, GruArch, TransformerArch] = LstmArch()
    bptt: int = 200
    training_procedure: TrainingProcedure = TrainingProcedure()
    config_version: str = CONFIG_VERSION

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION:
            raise TypeError(f'Trying to deserealize '
                            f'CONFIG_VERSION {self.config_version} '
                            f'to CONFIG_VERSION {CONFIG_VERSION} object')

    def get_arch_class(self) -> Union[Type[AWD_LSTM], Type[GRU], Type[Transformer]]:
        if isinstance(self.arch, LstmArch):
            return AWD_LSTM
        elif isinstance(self.arch, GruArch):
            return GRU
        elif isinstance(self.arch, TransformerArch):
            return Transformer
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")


def create_comet_experiment(run_id: str):
    experiment = Experiment()
    experiment.set_name(run_id)
    return experiment


class ExperimentRun:
    def __init__(self, run_id: str, config: LMTrainingConfig, device_options: DeviceOptions, comet_experiment: Optional):
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


@dataclass
class LMTrainingMetrics(object):
    bin_entropy: float
    training_time_minutes_per_epoch: int
    n_epochs: int
    best_epoch: int
