from typing import Optional, Dict, Type, Union

import jsons

from langmodels.lmconfig.datamodel import PrepFunction, LMTrainingConfig, LMTrainingMetrics, PrepFunctionOptions


def prep_function_serializer(prep_function: PrepFunction, **kwargs):
    return {'callable': prep_function.callable.__name__,
            'params': prep_function.params,
            'options':  jsons.dump(prep_function.options)}


def prep_function_deserializer(dct: Dict, cls: Type[PrepFunction], **kwargs) -> PrepFunction:
    import dataprep.api.corpus as api
    return cls(
        callable=getattr(api, dct['callable']),
        params=dct['params'],
        options=jsons.load(dct['options'], PrepFunctionOptions)
    )


jsons.set_serializer(prep_function_serializer, PrepFunction)
jsons.set_deserializer(prep_function_deserializer, cls=PrepFunction)


def dump_config(config: LMTrainingConfig, file: Optional[str]=None) -> str:
    str = jsons.dumps(config)
    if file:
        with open(file, 'w') as f:
            f.write(str)
    return str


def load_config_from_string(s: str) -> Union[LMTrainingConfig, LMTrainingMetrics]:
    return jsons.loads(s, Union[LMTrainingConfig, LMTrainingMetrics], strict=True)


def load_config_from_file(file: str) -> Union[LMTrainingConfig, LMTrainingMetrics]:
    with open(file, 'r') as f:
        s = f.read().replace('\n', '')
    return load_config_from_string(s)


def read_value_from_file(file: str, value_type):
    with open(file, 'r') as f:
        res = f.read().rstrip('\n')
    return value_type(res)
