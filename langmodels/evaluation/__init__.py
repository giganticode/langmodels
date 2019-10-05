from langmodels.evaluation.evaluation import evaluate_model_on_file, evaluate_model_on_string, evaluate_model_on_path
from langmodels.evaluation.visualization import write_html
from langmodels.model import TrainedModel


def evaluate(baseline_model: TrainedModel, target_model: TrainedModel, file: str):
    evaluation_result_baseline = evaluate_model_on_file(baseline_model, file)
    evaluation_result_target = evaluate_model_on_file(target_model, file)
    output_file = f'{file}.html'
    write_html([evaluation_result_baseline, evaluation_result_target], output_file)


__all__ = [evaluate_model_on_file, evaluate_model_on_string, evaluate_model_on_path, evaluate]
