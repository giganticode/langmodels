import logging
from dataclasses import dataclass

import torch
from typing import Union, Optional

from torch import cuda


logger = logging.getLogger(__name__)


class CudaNotAvailable(Exception):
    pass


def get_device(force_use_cpu: bool = False) -> Union[int, str]:
    return cuda.current_device() if cuda.is_available() and not force_use_cpu else 'cpu'


def get_map_location(force_use_cpu: bool):
    if force_use_cpu:
        map_location = lambda storage, loc: storage
        logger.debug("Using CPU for inference")
    elif cuda.is_available():
        map_location = torch.device('cuda:0')
        logger.debug("Using GPU for inference")
    else:
        map_location = lambda storage, loc: storage
        logger.info("Cuda not available. Falling back to using CPU.")
    return map_location


@dataclass(frozen=True)
class DeviceOptions(object):
    fallback_to_cpu: bool = False
    non_default_device_to_use: Optional[int] = None

    def get_device_id(self) -> Union[str, torch.device]:
        if cuda.is_available():
            device_id = self.non_default_device_to_use or cuda.current_device()
            return torch.device('cuda', device_id)
        elif self.fallback_to_cpu:
            return "cpu"
        else:
            raise CudaNotAvailable("Cuda not available")