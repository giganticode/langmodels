import os
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Union, Any, List, Dict

from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import AWD_LSTM, Transformer, TransformerXL, Activation


CONFIG_VERSION = '1.0.0'

HOME = os.environ['HOME']

PATH_TO_TRAINED_MODELS = os.environ['PATH_TO_TRAINED_MODELS'] if 'PATH_TO_TRAINED_MODELS' in os.environ else os.path.join(HOME, 'modelzoo')


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
    init_lr: float = 0.01
    mult_coeff: float = 0.5
    max_epochs: int = 50
    max_lr_reduction_times: int = 6


@dataclass(frozen=True)
class EarlyStop(object):
    patience: int = 3


@dataclass(frozen=True)
class CosineLRSchedule(TrainingSchedule):
    max_lr: float = 1e-4
    cyc_len: int = 5
    max_epochs: int = 50
    early_stop: EarlyStop = EarlyStop()


@dataclass(frozen=True)
class Corpus(object):
    path: str
    extensions: str  # in format "py" or "java|c|py"


PrepCallable = Callable[..., PreprocessedCorpus]
ParametrizedPrepCallable = Callable[[Corpus], PreprocessedCorpus]


@dataclass(frozen=True)
class PrepFunction(object):
    callable: PrepCallable
    params: List = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def apply(self) -> ParametrizedPrepCallable:
        def prep_corpus(corpus: Corpus, **kwargs) -> PreprocessedCorpus:
            return self.callable(corpus.path, *self.params, **self.options, **kwargs,
                                 calc_vocab=True, extensions=corpus.extensions)

        return prep_corpus


@dataclass(frozen=True)
class LstmArch(object):
    bidir: bool = False
    qrnn: bool = False
    emb_sz: int = 300
    n_hid: int = 650
    n_layers: int = 3
    adam_betas: Tuple[float, float] = (0.7, 0.99)
    clip: float = 0.3
    reg_fn: RegFn = RegFn()
    drop: Dropouts = Dropouts()
    tie_weights: bool = True
    out_bias: bool = True


@dataclass(frozen=True)
class TransformerArch(object):
    ctx_len: int = 512
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_head: int = 64
    d_inner: int = 3072
    resid_p: float = 0.1
    attn_p: float = 0.1
    ff_p: float = 0.1
    embed_p: float = 0.1
    output_p: float = 0.
    bias: bool = True
    scale: bool = True
    act: Activation = Activation.GeLU
    double_drop: bool = False
    tie_weights: bool = True
    out_bias: bool = False
    mask: bool = True


@dataclass(frozen=True)
class TrainingProcedure(object):
    schedule: RafaelsTrainingSchedule = RafaelsTrainingSchedule()
    weight_decay: float = 1e-6


@dataclass(frozen=True)
class Gpu(object):
    fallback_to_cpu: bool = False
    non_default_device_to_use: Optional[int] = None


@dataclass(frozen=True)
class LMTrainingConfig(object):
    base_model: Optional[str]
    bs: int
    corpus: Corpus
    prep_function: PrepFunction
    arch: Union[LstmArch, TransformerArch]
    bptt: int
    training_procedure: TrainingProcedure = TrainingProcedure()
    config_version: str = CONFIG_VERSION

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION:
            raise TypeError(f'Trying to deserealize '
                            f'CONFIG_VERSION {self.config_version} '
                            f'to CONFIG_VERSION {CONFIG_VERSION} object')

    def get_arch_class(self) -> Union[AWD_LSTM, Transformer, TransformerXL]:
        if isinstance(self.arch, LstmArch):
            return AWD_LSTM
        elif isinstance(self.arch, TransformerArch):
            return Transformer
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")


class Run:
    def __init__(self, config: LMTrainingConfig, gpu: Gpu):
        self.config = config
        self.gpu = gpu
        self.id = self._generate_run_id()

    @classmethod
    def with_config(cls, config: LMTrainingConfig, gpu: Gpu = Gpu()):
        return cls(config, gpu)

    def _generate_run_id(self) -> str:
        name_parts = []
        if self.config.base_model:
            name_parts.append([os.path.basename(self.config.base_model)])

        dataset = os.path.basename(self.config.corpus.path)
        prep_func_param = self.config.prep_function.params[0]
        n_layers = self.config.arch.n_layers
        n_hid = self.config.arch.n_hid

        import datetime
        time_now = datetime.datetime.now()
        timestamp = f"{time_now:%y%m%d.%H%M%S}"

        name_parts.append([dataset, str(prep_func_param), str(n_layers), str(n_hid), timestamp])

        return "_-_".join(map(lambda p: "_".join(p), name_parts))

    @property
    def path_to_trained_model(self):
        return os.path.join(PATH_TO_TRAINED_MODELS, self.id)
