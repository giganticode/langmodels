import colorsys
from math import log

from langmodels.evaluation.metrics import entropy_to_probability

MIN_HUE = 0.0
MAX_HUE = 0.33

LIGHT = 0.4
SATURATION = 0.7

EPS = 0.05
NEUTRAL_RANK = 4
NEUTRAL_PROBABILITY = 0.25


def normalize_rank(rank: int) -> float:
    """
    >>> normalize_rank(1)
    1.0

    >>> normalize_rank(2)
    0.41421356237309515

    >>> normalize_rank(4)
    0.0

    >>> normalize_rank(10)
    -0.3675444679663241

    >>> normalize_rank(100)
    -0.8

    """
    assert rank > 0

    return rank ** log(0.5, NEUTRAL_RANK) * 2 - 1


def normalize_entropy(entropy: float) -> float:
    """
    >>> normalize_entropy(0.0)
    1.0

    >>> normalize_entropy(0.83)
    0.5000389892858181

    >>> normalize_entropy(2.0)
    0.0

    >>> normalize_entropy(100.0)
    -0.9999999999999982

    >>> normalize_entropy(1e+6)
    -1.0

    """
    assert entropy >= 0.0

    prob = entropy_to_probability(entropy)
    result = prob ** log(0.5, NEUTRAL_PROBABILITY) * 2 - 1

    assert -1.0 <= result <= 1.0

    return result


def get_hex_color(metric: float) -> str:
    """
    >>> get_hex_color(0.0)
    '000000'

    >>> get_hex_color(0.09)
    'a2ad1f'

    >>> get_hex_color(0.1)
    'a1ad1f'

    >>> get_hex_color(-0.1)
    'ad9e1f'

    >>> get_hex_color(1.0)
    '21ad1f'

    >>> get_hex_color(-1.0)
    'ad1f1f'
    """
    assert -1.0 <= metric <= 1.0

    if abs(metric) < EPS:
        return '000000'

    neutral = (MAX_HUE + MIN_HUE) / 2.0
    span = (MAX_HUE - MIN_HUE) / 2.0

    rgb_normalized = colorsys.hls_to_rgb(neutral + span * metric, LIGHT, SATURATION)
    rgb = tuple(round(255 * c) for c in rgb_normalized)

    return ''.join(f'{i:02x}' for i in rgb)


def improvement_metric(before: float, after: float) -> float:
    assert -1.0 <= before <= 1.0
    assert -1.0 <= after <= 1.0

    return (after - before) / 2.0
