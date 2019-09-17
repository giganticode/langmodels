from torch import cuda


def get_device(force_use_cpu: bool= False) -> int:
    return cuda.current_device() if cuda.is_available() and not force_use_cpu else 'cpu'