import os

import psutil
import time
from collections import defaultdict

from typing import DefaultDict, Dict, List


class TimeMeasurer(object):
    def __init__(self):
        self._times: DefaultDict[str, List[float]] = defaultdict(list)
        self._in_progress: Dict[str, float] = {}

    def tick(self, what: str):
        if what in self._in_progress:
            raise ValueError(f"Tick has been already called for: {what}")

        self._in_progress[what] = time.perf_counter()

    def tock(self, what: str):
        if what not in self._in_progress:
            raise ValueError(f"Tick has not been called for: {what}")

        self._times[what].append(time.perf_counter() - self._in_progress[what])
        del self._in_progress[what]

    def totals(self):
        return {k: sum(v) for k, v in self._times.items()}


def get_cpu_memory_used_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 2 ** 20