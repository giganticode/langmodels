import os

import time
from pathlib import Path

import jsons
import matplotlib
from matplotlib import pyplot as plt

from typing import List, Tuple

from langmodels import repository
from langmodels.evaluation import evaluate_model_on_path
from langmodels.evaluation.customization import each_token_type_separately
from langmodels import project_dir


def get_run_name(path_basename: str) -> str:
    return path_basename + "_" + "_".join(time.ctime().split(" "))


def plot(entropies_list: List[Tuple[List[float], str]], dir: str, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel("Context")
    plt.ylabel("Increase in loss (times)")
    plt.title(title)
    for entropies, token_type in entropies_list:
        entropies = entropies[5:]
        ax.plot(range(5, len(entropies)+5), entropies, label=token_type)

    ax.legend(loc='upper right')
    # ax.set_yscale('log')
    plt.savefig(os.path.join(dir, title))
    plt.close()


def run_and_plot():
    m = repository.load_default_model()

    path = "/path/to/dataset"
    result = evaluate_model_on_path(m, Path(path), max_context_allowed=200, token_type_subsets=each_token_type_separately())

    matplotlib.use('Agg')

    run_name = get_run_name(os.path.basename(path))
    dir = os.path.join(project_dir, 'langmodels', 'figures', run_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    for scenario, summary in result.items():
        title = scenario.type_subset.short_summary
        plot(list(map(lambda x: x[0], summary.of_context_length)), dir, title)
        with open(os.path.join(dir, f'{title}.json'), 'w') as f:
            f.write(jsons.dumps(summary))


def relative_entropies(entropies: List[float]):
    smallest = min(entropies)
    return [(x - smallest) / smallest for x in entropies]


def plot_saved_ones(path: str, filenames: List[str]):
    matplotlib.use('Agg')
    results_for_each_token_type = []
    for filename in filenames:
        with open(os.path.join(path, filename), 'r') as f:
            obj = jsons.loads(f.read())
            entropies = list(map(lambda x: x[0], obj['of_context_length']))
            entropies = relative_entropies(entropies)
            results_for_each_token_type.append((entropies, os.path.splitext(filename)[0]))
    plot(results_for_each_token_type, path, 'MultipleTypes')


if __name__ == '__main__':
    # run_and_plot()
	
    plot_saved_ones(os.path.join(project_dir, 'langmodels/figures/test_Wed_Mar__4_00_06_41_2020'), [
        # 'ClosingBracket.json',
        # 'Semicolon.json',
        # 'ClosingCurlyBracket.json',
        # 'Comment.json',
        'KeyWord.json',
        'splitContainer.json',
        'ParsedToken.json',
    ])
