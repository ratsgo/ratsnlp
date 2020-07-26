import csv
import logging
from transformers import InputExample


logger = logging.getLogger(__name__)


class Corpus:

    @classmethod
    def _read_corpus(cls, input_file, quotechar=None):
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    def get_train_examples(self, data_path):
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(self._read_corpus(data_path), "train")

    def get_dev_examples(self, data_path):
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(self._read_corpus(data_path), "dev")

    def get_test_examples(self, data_path):
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(self._read_corpus(data_path), "test")

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def tfds_map(self, example):
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example


class NsmcCorpus(Corpus):

    @classmethod
    def _read_corpus(cls, input_file, quotechar='"'):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = None if set_type == "test" else line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


CORPUS_FACTORY = {
    "nsmc": NsmcCorpus,
}