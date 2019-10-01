import os
import sys
from dataclasses import dataclass

import re
from jinja2 import Template
from typing import List, Tuple

from langmodels import project_dir
from langmodels.evaluation.common import EvaluationResult, FullWordIterator
from langmodels.evaluation.evaluation import zip_subwords
from langmodels.evaluation.metrics import bin_entropy, full_token_mrr, DEFAULT_N_MODEL_SUGGESTIONS
from langmodels.evaluation.normalization import get_hex_color, normalize_entropy, normalize_rank, improvement_metric


@dataclass
class SubtokenToRender(object):
    text: str
    entropy1_color: str
    entropy2_color: str
    entropy_diff_color: str
    entropy1_tooltip: str
    entropy2_tooltip: str
    entropy_diff_tooltip: str


@dataclass
class TokenToRender(object):
    mrr1_color: str
    mrr2_color: str
    mrr_diff_color: str
    mrr1_tooltip: str
    mrr2_tooltip: str
    mrr_diff_tooltip: str
    subtokens: List[SubtokenToRender]


@dataclass
class LineToRender(object):
    leading_whitespace: str
    entropy1_color: str
    entropy2_color: str
    entropy_diff_color: str
    tokens: List[TokenToRender]


def format_subtoken(subword: str, ent1: float, ent2: float):
    improvement = improvement_metric(normalize_entropy(ent1), normalize_entropy(ent2))
    tooltip1 = f'{ent1:.2f}'
    tooltip2 = f'{ent2:.2f}'
    tooltip_diff = f'{tooltip1}->{tooltip2}'

    return SubtokenToRender(
        text=subword,
        entropy1_color=get_hex_color(normalize_entropy(ent1)),
        entropy2_color=get_hex_color(normalize_entropy(ent2)),
        entropy_diff_color=get_hex_color(improvement),
        entropy1_tooltip=tooltip1,
        entropy2_tooltip=tooltip2,
        entropy_diff_tooltip=tooltip_diff
    )


def format_token(full_word1: List[str], entropies1: List[float],
                 full_word2: List[str], entropies2: List[float],
                 rank1: int, rank2: int) -> TokenToRender:

    subtokens: List[SubtokenToRender] = []
    subwords, entropies = zip_subwords((full_word1, full_word2), (entropies1, entropies2))
    for subword, subword_entropies in zip(subwords, entropies):
        subtokens.append(format_subtoken(subword, subword_entropies[0], subword_entropies[1]))

    tooltip1 = rank1 if rank1 != sys.maxsize else f'>={DEFAULT_N_MODEL_SUGGESTIONS}'
    tooltip2 = rank2 if rank2 != sys.maxsize else f'>={DEFAULT_N_MODEL_SUGGESTIONS}'
    improvement = improvement_metric(normalize_rank(rank1), normalize_rank(rank2))
    return TokenToRender(
        mrr1_color=get_hex_color(normalize_rank(rank1)),
        mrr2_color=get_hex_color(normalize_rank(rank2)),
        mrr_diff_color=get_hex_color(improvement),
        mrr1_tooltip=tooltip1,
        mrr2_tooltip=tooltip2,
        mrr_diff_tooltip=f'{tooltip1}->{tooltip2}',
        subtokens=subtokens
    )


def format_results(evaluation_results: List[List[EvaluationResult]]) -> List[LineToRender]:
    result_to_render: List[LineToRender] = []
    for line_result1, line_result2 in zip(*evaluation_results):
        leading_whitespace = re.match("^\\s*", line_result1.text)[0]

        entropy1_color = get_hex_color(normalize_entropy(line_result1.aggregated_result[bin_entropy.__name__]))
        entropy2_color = get_hex_color(normalize_entropy(line_result2.aggregated_result[bin_entropy.__name__]))
        entropy_diff_color = get_hex_color(normalize_entropy(line_result2.aggregated_result[bin_entropy.__name__]))

        tokens: List[TokenToRender] = []
        for (full_word1, entropies1), (full_word2, entropies2), rank1, rank2 in zip(
                FullWordIterator(list(zip(line_result1.prep_text, line_result1.results[bin_entropy.__name__])),
                                 line_result1.prep_metadata.word_boundaries,
                                 agg=lambda a: tuple(list(i) for i in tuple(zip(*a)))),
                FullWordIterator(list(zip(line_result2.prep_text, line_result2.results[bin_entropy.__name__])),
                                 line_result2.prep_metadata.word_boundaries,
                                 agg=lambda a: tuple(list(i) for i in tuple(zip(*a)))),
                line_result1.results[full_token_mrr.__name__],
                line_result2.results[full_token_mrr.__name__]
        ):
            tokens.append(format_token(full_word1, entropies1, full_word2, entropies2, rank1, rank2))

        result_to_render.append(LineToRender(
            leading_whitespace=leading_whitespace,
            entropy1_color=entropy1_color,
            entropy2_color=entropy2_color,
            entropy_diff_color=entropy_diff_color,
            tokens=tokens
        ))
    return result_to_render


def write_html(evaluation_result: List[List[EvaluationResult]], output_file: str) -> None:
    if len(evaluation_result) != 2:
        raise ValueError('Evaluation results for 2 models must be passed')

    with open(os.path.join(project_dir, 'templates', 'evaluation.jinja2')) as f:
        template = Template(f.read())

    evaluation_to_render = format_results(evaluation_result)

    rendered = template.render(evaluation_to_render=evaluation_to_render)

    with open(output_file, 'w') as f:
        f.write(rendered)
    print(f'Output file: {output_file}')
