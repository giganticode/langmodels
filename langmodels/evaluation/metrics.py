import logging
from abc import ABC
from typing import Mapping, List

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
        tqdmed_batched_losses = tqdm(total=batched_token_loader.n_files, unit='files')
        all_structure_batches: List[CodeBaseStructure] = [CodeBaseStructure([]) for i in range(batched_token_loader.batch_size)]
        for entropies_batch, file_structure_batch, prepped_tokens_batch in calculate_losses(model, batched_token_loader):
            for entropies, prepped_tokens in zip(entropies_batch, prepped_tokens_batch):
                cur_batch_evaluation: EvaluationResultAccumulator = Entropy.sequence_entropy(entropies, prepped_tokens, evaluation_options, full_tokens)
                total_evaluation = total_evaluation.merge(cur_batch_evaluation)
            all_structure_batches = [all_structure_batches[i].merge(file_structure) for i, file_structure in enumerate(file_structure_batch)]
            Entropy.update_progress_bar(tqdmed_batched_losses, total_evaluation, all_structure_batches)
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
    def update_progress_bar(tqdmed_batched_losses: tqdm,
                            total_evaluation: EvaluationResultAccumulator, file_structure_batch: List[CodeBaseStructure]) -> None:
        prep_file_estimate = int(sum(map(lambda cs: len(cs.snippets), file_structure_batch)))
        tqdmed_batched_losses.update(prep_file_estimate - tqdmed_batched_losses.n)

        tqdmed_batched_losses.set_postfix(ordered_dict=total_evaluation.build().total())


metric_name_to_function: Mapping[str, Metric] = {
    Entropy.__name__: Entropy()
}