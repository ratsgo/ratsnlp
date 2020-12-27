import os
import time
import torch
import random
import logging
from Korpora import Korpora
from filelock import FileLock
from dataclasses import dataclass
from typing import List, Optional
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from ratsnlp.nlpbook.search.arguments import SearchTrainArguments

logger = logging.getLogger(__name__)


@dataclass
class SearchExample:
    question: str
    passage: str
    group: Optional[int] = None


@dataclass
class SearchFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None


@dataclass
class SearchPositivePair:
    question_features: SearchFeatures
    passage_features: SearchFeatures


class KoreanChatbotDataCorpus:

    def get_examples(self, data_path):
        corpus = Korpora.load(
            "korean_chatbot_data",
            root_dir=data_path,
            force_download=False,
        ).train
        # korean chatbot data는 전체 데이터가 11823건
        # unique한 쿼리가 11823건, 답변이 7779건이므로 답변 기준으로 그룹을 만든다
        group = {answer: idx for idx, answer in enumerate(set([el.pair for el in corpus]))}
        examples = []
        for el in corpus:
            example = SearchExample(
                question=el.text,
                passage=el.pair,
                group=group[el.pair]
            )
            examples.append(example)
        return examples


def _convert_examples_to_search_features(
        examples: List[SearchExample],
        tokenizer: PreTrainedTokenizer,
        args: SearchTrainArguments,
):

    logger.info(
        "tokenize sentences, it could take a lot of time..."
    )
    start = time.time()
    question_batch_encoding = tokenizer(
        [example.question for example in examples],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    passage_batch_encoding = tokenizer(
        [example.passage for example in examples],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    logger.info(
        "tokenize sentences [took %.3f s]", time.time() - start
    )

    features = defaultdict(list)
    for i in range(len(examples)):
        question_inputs = {k: question_batch_encoding[k][i] for k in question_batch_encoding}
        question_features = SearchFeatures(**question_inputs)
        passage_inputs = {k: passage_batch_encoding[k][i] for k in passage_batch_encoding}
        passage_features = SearchFeatures(**passage_inputs)
        features[examples[i].group].append([question_features, passage_features])

    for group_id in list(features.keys())[:5]:
        logger.info("*** Example group id: %s ***" % (group_id))
        for all_features in features[group_id]:
            q_features, p_features = all_features
            question_tokens = tokenizer.convert_ids_to_tokens(q_features.input_ids)
            passage_tokens = tokenizer.convert_ids_to_tokens(p_features.input_ids)
            positive_pair = " ".join([t for t in question_tokens if t != tokenizer.pad_token]) + "/" \
                            + " ".join([t for t in passage_tokens if t != tokenizer.pad_token])
            logger.info("positive pair tokens: %s" % (positive_pair))
            logger.info("question features: %s" % q_features)
            logger.info("passage features: %s" % p_features)

    return features


def search_train_collator(positive_pairs):
    features = defaultdict(list)
    for positive_pair in positive_pairs:
        for q_k, q_v in vars(positive_pair.question_features).items():
            features[f"question_{q_k}"].append(q_v)
        for p_k, p_v in vars(positive_pair.passage_features).items():
            features[f"passage_{p_k}"].append(p_v)
    labels = [el for el in range(len(positive_pairs))]
    all_features = {k: torch.LongTensor(v) for k, v in {**features, "labels": labels}.items()}
    return all_features


class SearchDataset(Dataset):

    def __init__(
            self,
            args: SearchTrainArguments,
            tokenizer: PreTrainedTokenizer,
            corpus: KoreanChatbotDataCorpus,
            mode: Optional[str] = "train",
            convert_examples_to_features_fn=_convert_examples_to_search_features,
    ):
        if corpus is not None:
            self.corpus = corpus
        else:
            raise KeyError("corpus is not valid")
        if not mode in ["train", "inference"]:
            raise KeyError(f"mode({mode}) is not a valid split name")
        else:
            self.mode = mode

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.downstream_corpus_root_dir,
            args.downstream_corpus_name,
            "cached_{}_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.downstream_corpus_name,
                "document-search",
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.downstream_corpus_root_dir}")
                examples = self.corpus.get_examples(args.downstream_corpus_root_dir)
                features = convert_examples_to_features_fn(
                    examples,
                    tokenizer,
                    args,
                )
                start = time.time()
                logging.info(
                    "Saving features into cached file, it could take a lot of time..."
                )
                torch.save(features, cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

            if self.mode == "train":
                self.features = features
            else:
                passages = set()
                passage_features = []
                for _, features_in_a_group in features.items():
                    for el in features_in_a_group:
                        _, passage_feature = el
                        passage = " ".join([str(el) for el in passage_feature.input_ids])
                        if passage not in passages:
                            passage_features.append(passage_feature)
                            passages.add(passage)
                self.features = passage_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        if self.mode == "train":
            # i번째 그룹에서 포지티브 페어를 랜덤으로 하나 뽑는다
            all_features = random.sample(self.features[i], 1)[0]
            question_features, passage_features = all_features
            return SearchPositivePair(
                question_features=question_features,
                passage_features=passage_features,
            )
        else:
            return self.features[i]