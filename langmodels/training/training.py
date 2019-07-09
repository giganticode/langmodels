import sys

import dataprep.api.corpus as api
from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import Vocab, TextList, NumericalizeProcessor, OpenFileProcessor, TokenizeProcessor, Tokenizer, \
    BaseTokenizer, TextLMDataBunch, language_model_learner, AWD_LSTM
from torch import cuda
from typing import List


class MyBaseTokenizer(BaseTokenizer):
    def tokenizer(self, t: str) -> List[str]:
        t = t.rstrip('\n')
        return super().tokenizer(t)


if __name__ == '__main__':
    prep_corpus:PreprocessedCorpus = api.bpe('/home/hlib/dev/yahtzee', '10k', calc_vocab=True)
    vocab = Vocab(['`unk'] + list(prep_corpus.load_vocab().keys()))
    tokenizer = Tokenizer(tok_func=MyBaseTokenizer, pre_rules=[], post_rules=[], special_cases=[], n_cpus=1)
    processors = [
        OpenFileProcessor(),
        TokenizeProcessor(tokenizer=tokenizer, include_bos=False, include_eos=False),
        NumericalizeProcessor(vocab=vocab, max_vocab=sys.maxsize, min_freq=0)
    ]
    text_list:TextLMDataBunch = TextList.from_folder(prep_corpus.path_to_prep_dataset, extensions=['.prep'], processor=processors)\
        .split_by_rand_pct()\
        .label_for_lm()\
        .databunch(bs=4, device=cuda.current_device())
    learn = language_model_learner(text_list, AWD_LSTM, drop_mult=0.5)
    learn.fit(10, 1e-2)
