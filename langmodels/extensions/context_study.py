import os
import time
from pathlib import Path

import jsons
import matplotlib
from matplotlib import pyplot as plt

from typing import List

from langmodels import repository
from langmodels.evaluation import evaluate_model_on_path
from langmodels.evaluation.customization import each_token_type_separately
from langmodels import project_dir


def get_run_name(path_basename: str) -> str:
    return path_basename + "_" + "_".join(time.ctime().split(" "))


def plot(entropies: List[float], dir: str, title: str):
    plt.xlabel("Context")
    plt.ylabel("Entropy")
    plt.title(title)
    plt.plot(range(len(entropies)), entropies)

    plt.savefig(os.path.join(dir, title))
    plt.close()


if __name__ == '__main__':
    m = repository.load_default_model()

    path = "/path/to/dataset"
    result = evaluate_model_on_path(m, Path(path), max_context_allowed=200, token_type_subsets=each_token_type_separately())

    matplotlib.use('TkAgg')

    run_name = get_run_name(os.path.basename(path))
    dir = os.path.join(project_dir, 'langmodels', 'figures', run_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    for scenario, summary in result.items():
        title = scenario.type_subset.short_summary
        plot(list(map(lambda x: x[0], summary.of_context_length)), dir, title)
        with open(os.path.join(dir, f'{title}.json'), 'w') as f:
            f.write(jsons.dumps(summary))