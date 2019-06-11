import argparse

import dataprep
from typing import List, Tuple, Callable

from langmodels.inference.model import TrainedModel


def get_entopy_for_each_line(trained_model: TrainedModel, file: str, entropy_aggregator: Callable[[List[float]], float], verbose: bool = False) -> List[float]:
    prep_lines_and_entropies: List[Tuple[List[str], List[float], float]] = []
    with open(file, 'r') as f:
        for line in f:
            prep_line = dataprep.bpe(line, trained_model.get_bpe_codes_id(), extension="java", **trained_model.get_prep_params())
            entropies = trained_model.get_entropies_for_next(prep_line)
            line_entropy = entropy_aggregator(entropies)
            prep_lines_and_entropies.append((prep_line, entropies, line_entropy))
        if verbose:
            for prep_line, entropies, line_entropy in prep_lines_and_entropies:
                print(f'{[(prep_token, token_entropy) for prep_token, token_entropy in zip(prep_line, entropies)]}')
                print(line_entropy)
                print("=============")
    return list(zip(*prep_lines_and_entropies))[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', action='store', help=f'Path to file for which entropies are to be calculated.')
    parser.add_argument('--output-path', action='store', help='Path to file to which entropies are to be written.')
    parser.add_argument('--verbose', action='store_true', help='Write preprocessed lines and their entropies to stdout.')
    args = parser.parse_args()
    verbose = args.verbose or 'output_path' not in args

    model = TrainedModel.get_default_model()
    entropies = get_entopy_for_each_line(model, args.file, lambda lst: sum(lst) / len(lst) if lst else .0, verbose)
    if 'output_path' in args:
        with open(args.output_path, 'w') as f:
            for entropy in entropies:
                f.write(f'{entropy}\n')
        print(f'Entropies are written to {args.output_path}')