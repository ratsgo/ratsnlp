import os
import csv
import time
import torch
import logging
from filelock import FileLock
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast
from ratsnlp.nlpbook.generation.arguments import GenerationTrainArguments


logger = logging.getLogger("ratsnlp")


@dataclass
class GenerationExample:
    text: str


@dataclass
class GenerationFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    labels: Optional[List[int]] = None


class NsmcCorpus:

    def __init__(self):
        pass

    def _read_corpus(cls, input_file, quotechar='"'):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            _, review_sentence, sentiment = line
            sentiment = "긍정" if sentiment == "1" else "부정"
            text = sentiment + " " + review_sentence
            examples.append(GenerationExample(text=text))
        return examples

    def get_examples(self, data_root_path, mode):
        data_fpath = os.path.join(data_root_path, f"ratings_{mode}.txt")
        logger.info(f"loading {mode} data... LOOKING AT {data_fpath}")
        return self._create_examples(self._read_corpus(data_fpath))


def _convert_examples_to_generation_features(
        examples: List[GenerationExample],
        tokenizer: PreTrainedTokenizerFast,
        args: GenerationTrainArguments,
):

    logger.info(
        "tokenize sentences, it could take a lot of time..."
    )
    start = time.time()
    batch_encoding = tokenizer(
        [example.text for example in examples],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    logger.info(
        "tokenize sentences [took %.3f s]", time.time() - start
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = GenerationFeatures(**inputs, labels=batch_encoding["input_ids"][i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("sentence: %s" % (example.text))
        logger.info("tokens: %s" % (" ".join(tokenizer.convert_ids_to_tokens(features[i].input_ids))))
        logger.info("features: %s" % features[i])

    return features


class GenerationDataset(Dataset):

    def __init__(
            self,
            args: GenerationTrainArguments,
            tokenizer: PreTrainedTokenizerFast,
            corpus,
            mode: Optional[str] = "train",
            convert_examples_to_features_fn=_convert_examples_to_generation_features,
    ):
        if corpus is not None:
            self.corpus = corpus
        else:
            raise KeyError("corpus is not valid")
        if not mode in ["train", "val", "test"]:
            raise KeyError(f"mode({mode}) is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.downstream_corpus_root_dir,
            args.downstream_corpus_name,
            "cached_{}_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.downstream_corpus_name,
                args.downstream_task_name,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                corpus_path = os.path.join(
                    args.downstream_corpus_root_dir,
                    args.downstream_corpus_name,
                )
                logger.info(f"Creating features from dataset file at {corpus_path}")
                examples = self.corpus.get_examples(corpus_path, mode)
                tokenizer.pad_token = tokenizer.eos_token
                self.features = convert_examples_to_features_fn(
                    examples,
                    tokenizer,
                    args,
                )
                start = time.time()
                logger.info(
                    "Saving features into cached file, it could take a lot of time..."
                )
                torch.save(self.features, cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.corpus.get_labels()
