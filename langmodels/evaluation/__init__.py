from langmodels.evaluation.evaluation import run_evaluation
from langmodels.evaluation.metrics import bin_entropy, full_token_mrr
from langmodels.evaluation.visualization import write_html
from langmodels.model import TrainedModel


def evaluate(baseline_model: TrainedModel, target_model: TrainedModel, file: str):
    evaluation_result = run_evaluation([baseline_model, target_model], file, [bin_entropy, full_token_mrr])
    output_file = f'{file}.html'
    write_html(evaluation_result, output_file)
