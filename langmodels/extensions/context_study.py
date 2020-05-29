import os
from pathlib import Path
from typing import List, Tuple

import jsons
import matplotlib
import time
from matplotlib import pyplot as plt

from langmodels import project_dir
from langmodels import repository
from langmodels.evaluation import evaluate_on_path
from langmodels.model.context import ContextModifier
from langmodels.model.tokencategories import each_token_type_separately
from langmodels.util.misc import HOME


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

    path = os.path.join(HOME, "dev/raw_datasets/allamanis/java-small-test")
    result = evaluate_on_path(m, Path(path),
                              context_modification=ContextModifier(max_context_length=300),
                              token_categories=each_token_type_separately())

    matplotlib.use('Agg')

    run_name = get_run_name(os.path.basename(path))
    dir = os.path.join(project_dir, 'langmodels', 'figures', run_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    for scenario, summary in result.items():
        title = scenario.token_category.short_summary
        plot([(list(map(lambda x: x[0], summary.values_for_contexts)), title)], dir, title)
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