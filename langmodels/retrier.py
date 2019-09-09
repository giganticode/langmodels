import logging

from fastai.basic_train import Learner
from retrying import retry
from typing import Any

from fastai.callbacks import SaveModelCallback

logger = logging.getLogger(__name__)


class RetryingSaveModelCalback(SaveModelCallback):
    def __init__(self, learn: Learner, monitor: str = 'val_loss', mode: str = 'auto', every: str = 'improvement',
                 name: str = 'bestmodel'):
        super().__init__(learn, monitor, mode, every, name)

    def _retry_if_io_error(exception):
        """Return True if we should retry (in this case when it's an IOError), False otherwise"""
        exception_type_to_retry = IOError
        wil_retry =  isinstance(exception, exception_type_to_retry)
        if wil_retry:
            logger.warning(f"Caught exception: {repr(exception)}. Will retry after some time ...")
        else:
            logger.warning(f"Exception was raised: {repr(exception)}. Retrying only {exception_type_to_retry}")
        return wil_retry

    WAIT_MULTIPLIER = 30 * 1000
    WAIT_MAX = 3600 * 1000

    @retry(retry_on_exception=_retry_if_io_error, wait_exponential_multiplier=WAIT_MULTIPLIER, wait_exponential_max=WAIT_MAX)
    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        super().on_epoch_end(epoch, **kwargs)

    @retry(retry_on_exception=_retry_if_io_error, wait_exponential_multiplier=WAIT_MULTIPLIER, wait_exponential_max=WAIT_MAX)
    def on_train_end(self, **kwargs):
        super().on_train_end(**kwargs)