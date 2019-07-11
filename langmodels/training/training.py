import dataprep.api.corpus as api
import numpy as np
from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import Vocab, TextList, language_model_learner, AWD_LSTM, PreProcessor, Collection
from torch import cuda
from tqdm import tqdm


class Numericalizer(PreProcessor):
    def __init__(self, vocab: Vocab):
        super().__init__()
        self.vocab = vocab

    def process_one(self,item):
        with open(item, 'r') as f:
            prep_tokens = [token for line in f for token in line.rstrip('\n').split(' ')]
            return np.array(self.vocab.numericalize(prep_tokens), dtype=np.int64)

    def process(self, ds: Collection):
        ds.vocab = self.vocab
        ds.items = np.array([self.process_one(item) for item in tqdm(ds.items)])


if __name__ == '__main__':
    prep_corpus:PreprocessedCorpus = api.bpe('/home/hlib/dev/yahtzee', '10k', calc_vocab=True)
    vocab = Vocab(['`unk', '`pad'] + list(prep_corpus.load_vocab().keys()))

    print(f'Getting preprocessed corpus from {prep_corpus.path_to_prep_dataset}')
    text_list = TextList.from_folder(prep_corpus.path_to_prep_dataset, extensions=['.prep'],
                                                     processor=Numericalizer(vocab))
    print("Splitting into training/validation sets")
    split_list = text_list.split_by_rand_pct()

    print("Labeling for langmodeling")
    labelled_list = split_list.label_for_lm()

    print("Creating databunches")
    databunched = labelled_list.databunch(bs=16, device=cuda.current_device())


    learner = language_model_learner(databunched, AWD_LSTM, drop_mult=0.5)
    learner.fit(10, 1e-2)
    # learner.save()
