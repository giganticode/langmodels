from typing import Dict, Union, Optional, Type

import jsons

from langmodels.lmconfig.datamodel import LMTrainingConfig, LMTrainingMetrics


def dump_to_file(config: Union[LMTrainingConfig, LMTrainingMetrics], file: str) -> str:
    config_str = jsons.dumps(config)
    with open(file, 'w') as f:
        f.write(config_str)
    return config_str


def load_config_or_metrics_form_dict(s: Dict[str, str], tp:Type=Union[LMTrainingConfig, LMTrainingMetrics]) -> Union[LMTrainingConfig, LMTrainingMetrics]:
    return jsons.load(s, tp, strict=True)


def load_config_from_string(s: str, type_to_load: Optional[Type]=Union[LMTrainingConfig, LMTrainingMetrics]) \
        -> Union[LMTrainingConfig, LMTrainingMetrics]:
    return jsons.loads(s, type_to_load, strict=True)


def load_config_or_metrics_from_file(file: str, type_to_load: Optional[Type] = Union[LMTrainingConfig, LMTrainingMetrics]) \
        -> Union[LMTrainingConfig, LMTrainingMetrics]:
    with open(file, 'r') as f:
        s = f.read().replace('\n', '')
    return load_config_from_string(s, type_to_load)


def read_value_from_file(file: str, value_type):
    with open(file, 'r') as f:
        res = f.read().rstrip('\n')
    return value_type(res)
