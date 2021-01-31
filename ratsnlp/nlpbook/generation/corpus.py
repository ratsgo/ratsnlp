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
from ratsnlp.nlpbook.generation.tokenizer import MASK_TOKEN
from ratsnlp.nlpbook.generation.arguments import GenerationTrainArguments


logger = logging.getLogger(__name__)


@dataclass
class KoreanChatExample:
    question: str
    answer: str


@dataclass
class GenerationFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[List[int]] = None


class KoreanChatCorpus:

    @classmethod
    def _read_corpus(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter=","))[1:]

    def _create_examples(self, lines):
        examples = []
        for line in lines:
            question, answer, _ = line
            examples.append(KoreanChatExample(question=question, answer=answer))
        return examples


def _convert_chatbot_examples_to_generation_features(
        examples: List[KoreanChatExample],
        tokenizer: PreTrainedTokenizerFast,
        args: GenerationTrainArguments,
):

    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    logger.info(
        "tokenize sentences, it could take a lot of time..."
    )
    start = time.time()
    features = []
    for example in examples:
        question_token_ids = [tokenizer.bos_token_id] + tokenizer.encode(example.question) + [tokenizer.eos_token_id]
        answer_token_ids = tokenizer.encode(example.answer) + [tokenizer.eos_token_id]
        answer_length = args.max_seq_length - len(question_token_ids)
        if answer_length > 0:
            if len(question_token_ids) + len(answer_token_ids) > args.max_seq_length:
                answer_token_ids = answer_token_ids[:answer_length]
            token_type_ids = [0] * len(question_token_ids) + [1] * len(answer_token_ids)
            attention_mask = [1] * (len(question_token_ids + len(answer_token_ids)))
            label = [mask_token_id] * len(question_token_ids) + answer_token_ids
            if len(question_token_ids) + len(answer_token_ids) < args.max_seq_length:
                padding_length = args.max_seq_length - len(question_token_ids) - len(answer_token_ids)
                answer_token_ids += [tokenizer.pad_token_id] * padding_length
                token_type_ids += [0] * padding_length
                attention_mask += [0] * padding_length
                label += [tokenizer.pad_token_id] * padding_length
            feature = GenerationFeatures(
                input_ids=question_token_ids + answer_token_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                label=label,
            )
            features.append(feature)
    logger.info(
        "tokenize sentences [took %.3f s]", time.time() - start
    )

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("question: %s" % (example.question))
        logger.info("answer: %s" % (example.answer))
        logger.info("tokens: %s" % (" ".join(tokenizer.decode(features[i].input_ids))))
        logger.info("label: %s" % (" ".join(tokenizer.decode(features[i].label))))
        logger.info("features: %s" % features[i])

    return features


class GenerationDataset(Dataset):

    def __init__(
            self,
            args: GenerationTrainArguments,
            tokenizer: PreTrainedTokenizerFast,
            corpus,
            mode: Optional[str] = "train",
            convert_examples_to_features_fn=_convert_chatbot_examples_to_generation_features,
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
                self.features = convert_examples_to_features_fn(
                    examples,
                    tokenizer,
                    args,
                )
                start = time.time()
                logging.info(
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
