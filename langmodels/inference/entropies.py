import logging

import argparse
from typing import List, Union

from langmodels.evaluation.common import EvaluationResult, get_file_extension
from langmodels.evaluation.metrics import bin_entropy
from langmodels.model import TrainedModel
from langmodels.profiling import TimeMeasurer

logger = logging.getLogger(__name__)


time_measurer = TimeMeasurer()


def get_entropy_for_each_line(trained_model: TrainedModel,
                              file: str,
                              verbose: bool = False,
                              only_aggregated_entropies: bool = True) -> Union[List[List[float]],
                                                                               List[EvaluationResult]]:
    prep_lines_and_entropies: List[EvaluationResult] = []
    with open(file, 'r') as f:
        extension = get_file_extension(file)
        for line in f:
            prep_line, metadata = trained_model.prep_text(line, return_metadata=True, force_reinit_bpe_data=False,
                                                          extension=extension)
            res, agg_results = bin_entropy(trained_model, prep_line, metadata)
            prep_lines_and_entropies.append(EvaluationResult(
                text=line,
                prep_text=prep_line,
                prep_metadata=metadata,
                results={bin_entropy.__name__: res},
                aggregated_result={bin_entropy.__name__: agg_results}
            ))
        if verbose:
            for line_results in prep_lines_and_entropies:
                print(line_results.text)
                print(line_results.aggregated_result)
                print(f"{[(prep_token, token_entropy) for prep_token, token_entropy in zip(line_results.prep_text, line_results.results[bin_entropy.__name__])]}")
                print("=============")
    return list(map(lambda e: e.aggregated_result[bin_entropy.__name__], prep_lines_and_entropies)) if only_aggregated_entropies \
        else prep_lines_and_entropies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', action='store', help=f'Path to file for which entropies are to be calculated.')
    parser.add_argument('-o', '--output-path', action='store', help='Path to file to which entropies are to be written.')
    parser.add_argument('-c', '--cpu', action='store_true', help='Forse cpu usage for inference even if cuda-supported GPU is available.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Write preprocessed lines and their entropies to stdout.')
    args = parser.parse_args()
    verbose = args.verbose or not args.output_path

    time_measurer.tick('Model loading')
    model = TrainedModel.get_default_model(force_use_cpu=args.cpu)
    time_measurer.tock('Model loading')
    results = get_entropy_for_each_line(model, args.file, verbose)
    if args.output_path:
        with open(args.output_path, 'w') as f:
            for entropy in results:
                f.write(f'{entropy}\n')
        print(f'Entropies are written to {args.output_path}')

    if verbose:
        totals = time_measurer.totals()
        for what, total_time in totals.items():
            logger.debug(f'{what} took {total_time:.4f} s')
