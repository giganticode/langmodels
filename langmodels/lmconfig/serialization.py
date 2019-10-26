from inspect import signature
from typing import Optional, Dict, Type

import jsons

from langmodels.lmconfig.datamodel import PrepFunction, LMTrainingConfig


def prep_function_serializer(prep_function: PrepFunction, **kwargs):
    SERIALIZABLE_OPTIONS = [
        'no_str', 'no_com', 'no_spaces', 'no_unicode', 'no_case'
    ]
    callable_parameters = signature(prep_function.callable).parameters
    serialized_options = {k: (prep_function.options[k] if k in prep_function.options else False) for k in
                          SERIALIZABLE_OPTIONS if k in callable_parameters}
    return {'callable': prep_function.callable.__name__,
            'params': prep_function.params,
            'options': serialized_options}


def prep_function_deserializer(dct: Dict, cls: Type[PrepFunction], **kwargs) -> PrepFunction:
    import dataprep.api.corpus as api
    return cls(
        callable=getattr(api, dct['callable']),
        params=dct['params'],
        options=dct['options']
    )


jsons.set_serializer(prep_function_serializer, PrepFunction)
jsons.set_deserializer(prep_function_deserializer, cls=PrepFunction)


def dump_config(config: LMTrainingConfig, file: Optional[str]=None) -> str:
    str = jsons.dumps(config)
    if file:
        with open(file, 'w') as f:
            f.write(str)
    return str


def load_config_from_string(s: str) -> LMTrainingConfig:
    return jsons.loads(s, LMTrainingConfig, strict=True)


def load_config_from_file(file: str) -> LMTrainingConfig:
    with open(file, 'r') as f:
        s = f.read().replace('\n', '')
    return load_config_from_string(s)


def read_value_from_file(file: str, value_type):
    with open(file, 'r') as f:
        res = f.read().rsplit('\n')
    return value_type(res)
