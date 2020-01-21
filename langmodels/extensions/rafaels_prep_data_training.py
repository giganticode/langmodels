import importlib
importlib.import_module('comet_ml')

import os

from dataprep.util import to_literal_str, to_non_literal_str
from langmodels import project_dir

from langmodels.lmconfig.datamodel import GruArch
from dataclasses import dataclass

from dataprep.pipeline.vocab import calc_vocab, VOCAB_DICT_DELIM, VOCAB_FILENAME
from langmodels.lmconfig.datamodel import LMTrainingConfig
from langmodels.training.training import train


def data_to_langmodels_format(file: str, output_path: str) -> None:
    SEPARATOR = '@@'
    with open(file, 'r') as f:
        text = f.read()
    tokens = text.split(' ')
    prepped_tokens = []
    for token in tokens:
        if token.endswith(SEPARATOR):
            token = token[:-len(SEPARATOR)]
        else:
            token += '</t>'
        token = to_literal_str(token)
        prepped_tokens.append(token)
    prepped_text = ' '.join(prepped_tokens)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, os.path.basename(file)), 'w') as f:
        f.write(prepped_text)


@dataclass
class RafaelsPreprocessedCorpus():
    path_to_prep_dataset: str
    path: str
    path_to_vocab: str

    @staticmethod
    def create(prepped_path: str) -> 'RafaelsPreprocessedCorpus':
        return RafaelsPreprocessedCorpus(prepped_path, prepped_path, os.path.join(prepped_path, VOCAB_FILENAME))

    def load_vocab(self):
        words = {}
        with open(self.path_to_vocab, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                splits = line.split(VOCAB_DICT_DELIM)
                words[to_non_literal_str(splits[0])] = int(splits[1])
        return words


if __name__ == '__main__':
    TRAINING_FILE = os.path.join(project_dir, 'data/rafaels_data/java/java_training_slp_pre_enc_bpe_10000')
    VALIDATION_FILE = os.path.join(project_dir, 'data/rafaels_data/java/java_validation_slp_pre_enc_bpe_10000')
    TEST_FILE = os.path.join(project_dir, 'data/rafaels_data/java/java_test_slp_pre_enc_bpe_10000')

    PREPPED_PATH = os.path.join(project_dir, 'data/prepped_rafaels_data')

    data_to_langmodels_format(TRAINING_FILE, os.path.join(PREPPED_PATH, 'train'))
    data_to_langmodels_format(VALIDATION_FILE, os.path.join(PREPPED_PATH, 'valid'))
    data_to_langmodels_format(TEST_FILE, os.path.join(PREPPED_PATH, 'test'))

    prepped_training_path = os.path.join(PREPPED_PATH, 'train', os.path.basename(TRAINING_FILE))

    calc_vocab(path=prepped_training_path, file_iterator=[prepped_training_path], output_dir=PREPPED_PATH)

    train(training_config=LMTrainingConfig(
        corpus=RafaelsPreprocessedCorpus.create(PREPPED_PATH), arch=GruArch(emb_sz=512, n_hid=512, n_layers=1)))
