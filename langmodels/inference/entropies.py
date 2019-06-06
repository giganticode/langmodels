import argparse

import dataprep
from typing import List

from langmodels.inference.model import TrainedModel


def get_entopy_for_each_line(trained_model: TrainedModel, file: str, verbose: bool = False) -> List[float]:
    prep_lines_and_entropies = []
    with open(file, 'r') as f:
        for line in f:
            prep_line = dataprep.bpe(line, trained_model.get_bpe_codes_id())
            entropy = trained_model.get_entropy_for_next(prep_line) if len(prep_line) != 0 else .0
            prep_lines_and_entropies.append((prep_line, entropy))
        if verbose:
            for prep_line, entropy in prep_lines_and_entropies:
                print(prep_line)
                print(entropy)
                print("=============")
    return list(zip(*prep_lines_and_entropies))[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', action='store', help=f'Path to file for which entropies are to be calculated.')
    parser.add_argument('--output-path', action='store', help='Path to file to which entropies are to be written.')
    parser.add_argument('--verbose', action='store_true', help='Write preprocessed lines and their entropies to stdout.')
    args = parser.parse_args()
    print(args.verbose)
    verbose = args.verbose or 'output_path' not in args

    model = TrainedModel.get_default_model()
    entropies = get_entopy_for_each_line(model, args.file, verbose)
    if 'output_path' in args:
        with open(args.output_path, 'w') as f:
            for entropy in entropies:
                f.write(f'{entropy}\n')
        print(f'Entropies are written to {args.output_path}')