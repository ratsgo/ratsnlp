import os
import time
import torch
import logging
from filelock import FileLock
from typing import List, Optional, Union
from torch.utils.data.dataset import Dataset
from ratsnlp.nlpbook.arguments import Arguments
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import InputExample, InputFeatures, PreTrainedTokenizer


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


class DownstreamDataset(Dataset):

    args: Arguments
    corpus: Corpus
    features: List[InputFeatures]

    def __init__(
        self,
        args: Arguments,
        tokenizer: PreTrainedTokenizer,
        corpus: Corpus,
        limit_length: Optional[int] = None,
        mode: Optional[str] = "train",
        cache_dir: Optional[str] = None,
    ):
        if corpus is not None:
            self.corpus = corpus
        else:
            raise KeyError("corpus is not valid")
        if not mode in ["train", "val", "test"]:
            raise KeyError(f"mode({mode}) is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.downstream_corpus_dir,
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
                logger.info(f"Creating features from dataset file at {args.downstream_corpus_dir}")

                corpus_fpath = os.path.join(args.downstream_corpus_dir, f"{mode}.txt")
                if mode == "val":
                    examples = self.corpus.get_dev_examples(corpus_fpath)
                elif mode == "test":
                    examples = self.corpus.get_test_examples(corpus_fpath)
                else:
                    examples = self.corpus.get_train_examples(corpus_fpath)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = _convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    loss_type="classification",
                    label_list=self.corpus.get_labels(),
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

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.corpus.get_labels()


def _convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    loss_type: str = "classification",
    label_list: Optional[List[str]] = None,
):

    if max_length is None:
        max_length = tokenizer.max_len

    if loss_type == "classification":
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = {}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if loss_type == "classification":
            return label_map[example.label]
        elif loss_type == "regression":
            return float(example.label)
        raise KeyError(loss_type)

    labels = [label_from_example(example) for example in examples]

    logger.info(
        "tokenize sentences, it could take a lot of time..."
    )
    start = time.time()
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    logger.info(
        "tokenize sentences [took %.3f s]", time.time() - start
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("sentence: %s" % (example.text_a))
        logger.info("tokens: %s" % (" ".join(tokenizer.convert_ids_to_tokens(features[i].input_ids))))
        logger.info("label: %s" % (example.label))
        logger.info("features: %s" % features[i])

    return features


def get_dataloaders(corpus_class, tokenizer, args):
    if args.do_train:
        train_dataset = DownstreamDataset(
            args=args,
            corpus=corpus_class,
            tokenizer=tokenizer,
            mode="train",
            cache_dir=args.data_cache_dir
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_dataset, replacement=False),
            drop_last=False,
            num_workers=args.cpu_workers,
        )
    else:
        train_dataloader = None

    if args.do_eval:
        val_dataset = DownstreamDataset(
            args=args,
            corpus=corpus_class,
            tokenizer=tokenizer,
            mode="val",
            cache_dir=args.data_cache_dir
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(val_dataset),
            drop_last=False,
            num_workers=args.cpu_workers,
        )
    else:
        val_dataloader = None

    if args.do_predict:
        test_dataset = DownstreamDataset(
            args=args,
            corpus=corpus_class,
            tokenizer=tokenizer,
            mode="test",
            cache_dir=args.data_cache_dir
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(test_dataset),
            drop_last=False,
            num_workers=args.cpu_workers,
        )
    else:
        test_dataloader = None
    return train_dataloader, val_dataloader, test_dataloader
