import csv
import logging
from transformers import InputExample
from ratsnlp.nlpbook.dataset import Corpus


logger = logging.getLogger(__name__)


class KoreanChatCorpus(Corpus):

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
