import os
import logging
from ratsnlp.nlpbook.classification.corpus import ClassificationExample

logger = logging.getLogger(__name__)


class KorNLICorpus:

    def __init__(self):
        pass

    def _read_corpus(cls, input_file):
        corpus = open(input_file, "r", encoding="utf-8").readlines()
        return [line.strip().split("\t") for line in corpus]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a, text_b, label = line
            examples.append(ClassificationExample(text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_examples(self, data_path, mode):
        logger.info(f"loading {mode} data... LOOKING AT {data_path}")
        if mode == "train":
            multinli_train_data_fpath = os.path.join(data_path, "multinli.train.ko.tsv")
            multinli_train_data = self._create_examples(self._read_corpus(multinli_train_data_fpath), mode)
            snli_train_data_fpath = os.path.join(data_path, "snli_1.0_train.ko.tsv")
            snli_train_data = self._create_examples(self._read_corpus(snli_train_data_fpath), mode)
            data = multinli_train_data + snli_train_data
        elif mode == "val":
            valid_data_fpath = os.path.join(data_path, "xnli.dev.ko.tsv")
            data = self._create_examples(self._read_corpus(valid_data_fpath), mode)
        else:
            test_data_fpath = os.path.join(data_path, "xnli.test.ko.tsv")
            data = self._create_examples(self._read_corpus(test_data_fpath), mode)
        return data

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]

    @property
    def num_labels(self):
        return len(self.get_labels())
