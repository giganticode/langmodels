import logging
import os

import argparse
from typing import List, Tuple, Callable, Union, Dict

from langmodels.model import TrainedModel
from langmodels.profiling import TimeMeasurer

logger = logging.getLogger(__name__)

time_measurer = TimeMeasurer()


def get_entropy_for_each_line(trained_model: TrainedModel,
                              file: str,
                              entropy_aggregator: Callable[[List[float], List[int]], Union[float, List[float]]],
                              verbose: bool = False) -> Union[List[float], List[List[float]]]:
    prep_lines_and_entropies: List[Dict[str, Union[str, List[str], List[float], float]]] = []
    with open(file, 'r') as f:
        _, extension = os.path.splitext(file)
        for line in f:
            time_measurer.tick("Inference")
            prep_line, entropies, word_boundaries = trained_model.get_entropies_for_text(line, extension[1:])
            time_measurer.tock("Inference")
            line_entropy = entropy_aggregator(entropies, word_boundaries)
            prep_lines_and_entropies.append({
                'text': line,
                'prep_text': prep_line,
                'entropies': entropies,
                'line_entropy': line_entropy
            })
        if not verbose:
            for line in prep_lines_and_entropies:
                print(line['text'])
                print(line['line_entropy'])
                print(f"{[(prep_token, token_entropy) for prep_token, token_entropy in zip(line['prep_text'], line['entropies'])]}")
                print("=============")
    return list(map(lambda e: e['line_entropy'], prep_lines_and_entropies))


def subword_average(subword_entropies: List[float], word_boundaries: List[int]) -> float:
    return sum(subword_entropies) / len(subword_entropies) if subword_entropies else .0


def word_entropy_list(subword_entropies: List[float], word_boundaries: List[int]) -> List[float]:
    if not word_boundaries or word_boundaries[-1] != len(subword_entropies):
        raise ValueError(f"Word boundaries list should contain the index of the last word "
                         f"(or at least 0 if subword_entropies list is empty).\n"
                         f"However, the subword entropies list has {len(subword_entropies)} elements, and "
                         f"value {len(subword_entropies)} is not found in word boundaries list: {word_boundaries}")

    word_entropies = []
    for i in range(len(word_boundaries) - 1):
        word_start_index = word_boundaries[i]
        word_end_index = word_boundaries[i+1]
        word_entropies.append(sum(subword_entropies[word_start_index: word_end_index]))
    return word_entropies


def word_average(subword_entropies: List[float], word_boundaries: List[int]) -> float:
    word_entropies = word_entropy_list(subword_entropies, word_boundaries)
    if not word_entropies:
        return .0

    return sum(word_entropies) / len(word_entropies)


def parse_entropy_aggregator_value(entropy_aggregator_name: str) -> Callable[[List[float], List[int]], Union[float, List[float]]]:
    if entropy_aggregator_name == 'subtoken-average':
        return subword_average
    elif entropy_aggregator_name == 'full-token-average':
        return word_average
    elif entropy_aggregator_name == 'full-token-entropies':
        return word_entropy_list
    else:
        raise ValueError(f"Unknown value for entropy aggregator: {entropy_aggregator_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', action='store', help=f'Path to file for which entropies are to be calculated.')
    parser.add_argument('-o', '--output-path', action='store', help='Path to file to which entropies are to be written.')
    parser.add_argument('-e', '--entropy-aggregator', action='store', default='full-token-average',
                        help='Fuction to calculate entropy for the whole line from subtoken entropies. Possible values:\n'
                             '\'subtoken-average\' (default): average over all subtokens\' entropies \n'
                             '\'full-token-average\': average over all full-tokens\' entopies '
                             '(entropy of a full token is a sum of entopies of its subtokens to which a token was split during pre-processing) \n'
                             '\'full-token-entropies\': a list of full-token entropies (gives freedom to library\'s clients to compute line-entropy in their own way) \n')
    parser.add_argument('-c', '--cpu', action='store_true', help='Forse cpu usage for inference even if cuda-supported GPU is available.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Write preprocessed lines and their entropies to stdout.')
    args = parser.parse_args()
    verbose = args.verbose or not args.output_path

    time_measurer.tick('Model loading')
    model = TrainedModel.get_default_model(force_use_cpu=args.cpu)
    time_measurer.tock('Model loading')
    entropy_aggregator = parse_entropy_aggregator_value(args.entropy_aggregator)
    entropies = get_entropy_for_each_line(model, args.file, entropy_aggregator, verbose)
    if args.output_path:
        with open(args.output_path, 'w') as f:
            for entropy in entropies:
                f.write(f'{entropy}\n')
        print(f'Entropies are written to {args.output_path}')

    if verbose:
        totals = time_measurer.totals()
        for what, total_time in totals.items():
            logger.debug(f'{what} took {total_time:.4f} s')
