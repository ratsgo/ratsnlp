import os
import json
import logging
from ratsnlp.nlpbook.classification.corpus import ClassificationExample


logger = logging.getLogger("ratsnlp")


class KlueNLICorpus:

    def __init__(self):
        pass

    def _create_examples(self, data_path):
        examples = []
        data = json.load(open(data_path, "r"))
        for el in data:
            example = ClassificationExample(
                text_a=el["premise"],
                text_b=el["hypothesis"],
                label=el["gold_label"],
            )
            examples.append(example)
        return examples

    def get_examples(self, data_path, mode):
        if mode == "train":
            data_fpath = os.path.join(data_path, "klue_nli_train.json")
        else:
            data_fpath = os.path.join(data_path, "klue_nli_dev.json")
        logger.info(f"loading {mode} data... LOOKING AT {data_fpath}")
        examples = self._create_examples(data_fpath)
        return examples

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]

    @property
    def num_labels(self):
        return len(self.get_labels())



class KorNLICorpus:

    def __init__(self):
        pass

    def _create_examples(self, data_path):
        examples = []
        corpus = open(data_path, "r", encoding="utf-8").readlines()
        lines = [line.strip().split("\t") for line in corpus]
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
            multinli_train_data = self._create_examples(multinli_train_data_fpath)
            snli_train_data_fpath = os.path.join(data_path, "snli_1.0_train.ko.tsv")
            snli_train_data = self._create_examples(snli_train_data_fpath)
            examples = multinli_train_data + snli_train_data
        elif mode == "val":
            valid_data_fpath = os.path.join(data_path, "xnli.dev.ko.tsv")
            examples = self._create_examples(valid_data_fpath)
        else:
            test_data_fpath = os.path.join(data_path, "xnli.test.ko.tsv")
            examples = self._create_examples(test_data_fpath)
        return examples

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]

    @property
    def num_labels(self):
        return len(self.get_labels())
