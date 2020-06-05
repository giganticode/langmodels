import logging
from abc import ABC
from typing import Mapping

from tqdm import tqdm

from codeprep.preprocess.codestructure import CodeBaseStructure
from langmodels.evaluation.characteristics import characterize_token
from langmodels.evaluation.dataloader import BatchedTokenLoader
from langmodels.evaluation.options import EvaluationOptions
from langmodels.evaluation.result import EvaluationResultAccumulator
from langmodels.model._evaluator import calculate_losses
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
        tqdmed_batched_losses = tqdm(calculate_losses(model, batched_token_loader), total=batched_token_loader.estimated_n_batches())
        for entropies_batch, file_structure_batch, prepped_tokens_batch in tqdmed_batched_losses:
            for entropies, prepped_tokens in zip(entropies_batch, prepped_tokens_batch):
                cur_batch_evaluation: EvaluationResultAccumulator = Entropy.sequence_entropy(entropies, prepped_tokens, evaluation_options, full_tokens)
                total_evaluation = total_evaluation.merge(cur_batch_evaluation)

            Entropy.update_progress_bar(tqdmed_batched_losses, batched_token_loader, total_evaluation, file_structure_batch)
        tqdmed_batched_losses.close()
        return total_evaluation

    @staticmethod
    def sequence_entropy(entropies, prepped_tokens, evaluation_options, full_tokens) -> EvaluationResultAccumulator:
        assert prepped_tokens.is_complete()
        evaluation_result: EvaluationResultAccumulator = EvaluationResultAccumulator.empty(evaluation_options.characteristics, evaluation_options.metric_names)
        tokens = prepped_tokens.full_token_view(return_metadata=True) if full_tokens else prepped_tokens.sub_token_view(return_metadata=True)
        entropies_iterator = tokens.get_iterator(over=entropies, over_full_tokens=False, formatter=sum)
        for single_token_seq_elm, entropy in zip(tokens, entropies_iterator):
            token_characteristics = characterize_token(single_token_seq_elm, evaluation_options.characteristics, None)
            evaluation_result.add(Entropy.__name__, to_full_token_string(single_token_seq_elm.tokens, include_debug_tokens=True), token_characteristics, entropy)
        return evaluation_result

    @staticmethod
    def update_progress_bar(tqdmed_batched_losses: tqdm, batched_token_loader: BatchedTokenLoader,
                            total_evaluation: EvaluationResultAccumulator, file_structure_batch: CodeBaseStructure) -> None:
        tqdmed_batched_losses.update(sum(map(lambda cs: len(cs.snippets) - 1, file_structure_batch)))
        evaluation_result_to_display = total_evaluation.build().total()
        estimated_n_batches = batched_token_loader.estimated_n_batches()
        current_iteration = batched_token_loader.current_iteration
        estimated_iterations_left = estimated_n_batches - current_iteration
        progress_bar_needs_to_be_updated = (estimated_iterations_left & (estimated_iterations_left - 1) == 0
                                            or current_iteration & (current_iteration - 1) == 0)
        tqdmed_batched_losses.set_postfix(ordered_dict=evaluation_result_to_display)
        if progress_bar_needs_to_be_updated:
            tqdmed_batched_losses.total = estimated_n_batches
            tqdmed_batched_losses.refresh()


metric_name_to_function: Mapping[str, Metric] = {
    Entropy.__name__: Entropy()
}