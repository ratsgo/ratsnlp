import os
import time
import torch
import logging
from filelock import FileLock
from typing import List, Optional, Union
from torch.utils.data.dataset import Dataset
from transformers import InputExample, InputFeatures, PreTrainedTokenizer
from ratsnlp.arguments import PretrainedModelArguments, DownstreamDataArguments
from ratsnlp.corpus import Corpus, CORPUS_FACTORY


logger = logging.getLogger(__name__)


class DownstreamDataset(Dataset):

    model_args: PretrainedModelArguments
    data_args: DownstreamDataArguments
    corpus: Corpus
    features: List[InputFeatures]

    def __init__(
        self,
        model_args: PretrainedModelArguments,
        data_args: DownstreamDataArguments,
        tokenizer: PreTrainedTokenizer,
        corpus: Optional[Corpus] = None,
        limit_length: Optional[int] = None,
        mode: Optional[str] = "train",
        cache_dir: Optional[str] = None,
    ):
        if corpus is not None:
            self.corpus = corpus
        elif data_args.corpus_name in CORPUS_FACTORY.keys():
            self.corpus = CORPUS_FACTORY[data_args.corpus_name]()
        else:
            raise KeyError("corpus is not valid")
        if not model_args.loss_type in ["classification", "regression"]:
            raise KeyError(f"loss type is classification or regression. but current setting is {model_args.loss_type}")
        self.loss_type = model_args.loss_type
        if not mode in ["train", "dev", "test"]:
            raise KeyError(f"mode({mode}) is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else data_args.corpus_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, tokenizer.__class__.__name__, str(data_args.max_seq_length), data_args.corpus_name, data_args.task_name,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not data_args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {data_args.corpus_dir}")

                corpus_fpath = os.path.join(data_args.corpus_dir, f"{mode}.txt")
                if mode == "dev":
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
                    max_length=data_args.max_seq_length,
                    loss_type=self.loss_type,
                    label_list=self.corpus.get_labels() if self.loss_type == "classification" else None,
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


def get_datasets(corpus_class, tokenizer, args):
    pretrained_model_args, downstream_data_args, fine_tuning_args = args
    train_dataset = (
        DownstreamDataset(pretrained_model_args, downstream_data_args,
                          corpus=corpus_class, tokenizer=tokenizer, mode="train",
                          cache_dir=downstream_data_args.data_cache_dir)
        if fine_tuning_args.do_train
        else None
    )
    eval_dataset = (
        DownstreamDataset(pretrained_model_args, downstream_data_args,
                          corpus=corpus_class, tokenizer=tokenizer, mode="dev",
                          cache_dir=downstream_data_args.data_cache_dir)
        if fine_tuning_args.do_eval
        else None
    )
    test_dataset = (
        DownstreamDataset(pretrained_model_args, downstream_data_args,
                          corpus=corpus_class, tokenizer=tokenizer, mode="test",
                          cache_dir=downstream_data_args.data_cache_dir)
        if fine_tuning_args.do_predict
        else None
    )
    return train_dataset, eval_dataset, test_dataset
