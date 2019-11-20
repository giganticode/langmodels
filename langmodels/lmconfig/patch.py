from typing import Any, Dict

from langmodels.lmconfig.datamodel import LMTrainingConfig


def patch_config(config: LMTrainingConfig, patch: Dict[str, Any]) -> LMTrainingConfig:
    for path_to_param, new_value in patch.items():
        obj = config
        attrs = path_to_param.split(".")
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        getattr(obj, attrs[-1])
        setattr(obj, attrs[-1], new_value)
    return config
