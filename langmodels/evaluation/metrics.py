import logging
from abc import ABC
from typing import Mapping, List

from torch import Tensor
from tqdm import tqdm

from codeprep.preprocess.codestructure import CodeBaseStructure
from codeprep.preprocess.tokens import TokenSequence
from langmodels.evaluation.characteristics import characterize_token
from langmodels.evaluation.dataloader import BatchedTokenLoader
from langmodels.evaluation.options import EvaluationOptions
from langmodels.evaluation.result import EvaluationResultAccumulator
from langmodels.evaluation.evaluator import calculate_losses_for_batch
from langmodels.model.model import TrainedModel, retain_models_state, to_full_token_string

logger = logging.getLogger(__name__)


class Metric(ABC):
    def __call__(model: TrainedModel, batched_token_loader: BatchedTokenLoader,
                 evaluation_options: EvaluationOptions,
                 full_tokens: bool = True) -> EvaluationResultAccumulator:
        pass


class Entropy(Metric):
    @retain_models_state
    def __call__(self, model: TrainedModel, batched_token_loader: BatchedTokenLoader,
                evaluation_options: EvaluationOptions,
                full_tokens: bool = True) -> EvaluationResultAccumulator:
        total_evaluation: EvaluationResultAccumulator = EvaluationResultAccumulator.empty(evaluation_options.characteristics, list(metric_name_to_function.keys()))
        tqdmed_batched_losses = tqdm(total=batched_token_loader.n_files, unit='files')
        all_structure_batches: List[CodeBaseStructure] = [CodeBaseStructure.empty() for i in range(batched_token_loader.batch_size)]
        for losses_with_metadata_batch in calculate_losses_for_batch(model, batched_token_loader):
            for i, losses_with_metadata in enumerate(losses_with_metadata_batch):
                cur_batch_evaluation: EvaluationResultAccumulator = Entropy.sequence_entropy(
                    losses_with_metadata.losses.tolist(), losses_with_metadata.prepped_tokens,
                    evaluation_options, full_tokens
                )
                total_evaluation = total_evaluation.merge(cur_batch_evaluation)
                all_structure_batches[i].merge(losses_with_metadata.code_structure)
            Entropy.update_progress_bar(tqdmed_batched_losses, total_evaluation, all_structure_batches)
        tqdmed_batched_losses.close()
        return total_evaluation

    @staticmethod
    def sequence_entropy(entropies: List[float], prepped_tokens: TokenSequence, evaluation_options: EvaluationOptions, full_tokens: bool) -> EvaluationResultAccumulator:
        assert prepped_tokens.is_complete()
        evaluation_result: EvaluationResultAccumulator = EvaluationResultAccumulator.empty(evaluation_options.characteristics, evaluation_options.metric_names)
        tokens = prepped_tokens.full_token_view(return_metadata=True) if full_tokens else prepped_tokens.sub_token_view(return_metadata=True)
        entropies_iterator = tokens.get_iterator(over=entropies, over_full_tokens=False, formatter=sum)
        for single_token_seq_elm, entropy in zip(tokens, entropies_iterator):
            token_characteristics = characterize_token(single_token_seq_elm, evaluation_options.characteristics, None)
            full_token_string = to_full_token_string(single_token_seq_elm.tokens, include_debug_tokens=True) if single_token_seq_elm.is_complete() else single_token_seq_elm.tokens[0]
            evaluation_result.add(Entropy.__name__, full_token_string, token_characteristics, entropy)
        return evaluation_result

    @staticmethod
    def update_progress_bar(tqdmed_batched_losses: tqdm,
                            total_evaluation: EvaluationResultAccumulator, file_structure_batch: List[CodeBaseStructure]) -> None:
        prep_file_estimate = int(sum(map(lambda cs: len(cs.snippets), file_structure_batch)))
        tqdmed_batched_losses.update(prep_file_estimate - tqdmed_batched_losses.n)
        if tqdmed_batched_losses.total//32 & (tqdmed_batched_losses.total//32 - 1) == 0 or prep_file_estimate//32 & (prep_file_estimate//32 - 1) == 0:
            tqdmed_batched_losses.set_postfix(ordered_dict=total_evaluation.build().total())


metric_name_to_function: Mapping[str, Metric] = {
    Entropy.__name__: Entropy()
}