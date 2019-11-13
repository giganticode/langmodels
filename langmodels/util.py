from math import log

from torch import FloatTensor
from typing import Union


def to_binary_entropy(entropy: Union[float, FloatTensor]) -> Union[float, FloatTensor]:
    return entropy / log(2)