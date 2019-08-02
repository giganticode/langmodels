from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Union, Any, List, Dict

from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import AWD_LSTM, Transformer, TransformerXL, Activation

CONFIG_VERSION = '1.0.0'


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
class LearningRateCycle(object):
    n: int
    len: int
    mult: int


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
        def prep_corpus(corpus: Corpus, **kwargs):
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
    base_lr: float
    cycle: LearningRateCycle
    weight_decay: float = 1e-6
    early_stop: bool = True  ###


@dataclass(frozen=True)
class LMTrainingConfig(object):
    base_model: Optional[str]
    corpus: Corpus
    prep_function: PrepFunction
    arch: Union[LstmArch, TransformerArch]
    bs: int
    bptt: int
    training_procedure: TrainingProcedure
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
