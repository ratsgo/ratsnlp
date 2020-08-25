import os
import re
import time
import torch
import logging
from filelock import FileLock
from typing import List, Optional
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from ratsnlp.nlpbook.arguments import TrainArguments
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy


logger = logging.getLogger(__name__)
NER_CLS_TOKEN = "[CLS]"
NER_SEP_TOKEN = "[SEP]"
NER_PAD_TOKEN = "[PAD]"
NER_MASK_TOKEN = "[MASK]"
NER_PAD_ID = 2


@dataclass
class NERExample:
    text: str
    label: Optional[str] = None


@dataclass
class NERFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class NERCorpus:

    def __init__(self, args: TrainArguments):
        self.args = args

    def get_examples(self, data_path, mode):
        logger.info(f"loading {mode} data... LOOKING AT {data_path}")
        examples = []
        for line in open(data_path, "r", encoding="utf-8").readlines():
            text, label = line.split("\u241E")
            examples.append(NERExample(text=text, label=label))
        return examples

    def get_labels(self):
        label_map_path = os.path.join(self.args.downstream_model_dir, "label_map.txt")
        if not os.path.exists(label_map_path):
            logging.info("processing NER tag dictionary...")
            os.makedirs(self.args.downstream_model_dir, exist_ok=True)
            ner_tags = []
            regex_ner = re.compile('<(.+?):[A-Z]{3}>')
            train_corpus_path = os.path.join(self.args.downstream_corpus_dir, "train.txt")
            target_sentences = [line.split("\u241E")[1].strip()
                                for line in open(train_corpus_path, "r", encoding="utf-8").readlines()]
            for target_sentence in target_sentences:
                regex_filter_res = regex_ner.finditer(target_sentence)
                for match_item in regex_filter_res:
                    ner_tag = match_item[0][-4:-1]
                    if ner_tag not in ner_tags:
                        ner_tags.append(ner_tag)
            b_tags = [f"B-{ner_tag}" for ner_tag in ner_tags]
            i_tags = [f"I-{ner_tag}" for ner_tag in ner_tags]
            labels = [NER_CLS_TOKEN, NER_SEP_TOKEN, NER_PAD_TOKEN, NER_MASK_TOKEN, "O"] + b_tags + i_tags
            with open(label_map_path, "w", encoding="utf-8") as f:
                for tag in labels:
                    f.writelines(tag + "\n")
        else:
            labels = [tag.strip() for tag in open(label_map_path, "r", encoding="utf-8").readlines()]
        return labels

    @property
    def num_labels(self):
        return len(self.get_labels())


def _process_target_sentence(
        tokens: List[str],
        target_sentence: str,
        max_length: int,
        label_map: dict,
        cls_token_at_end: Optional[bool] = False,
):
    """
    target_sentence = "첫 회를 시작으로 <13일:DAT>까지 <4일간:DUR> 총 <4회:NOH>에 걸쳐 매 회 <2편:NOH>씩 총 <8편:NOH>이 공개될 예정이다."
    tokens = ['첫', '회', '##를', '시작으로', '13', '##일까지', '4', '##일간', '총', '4', '##회', '##에', '걸쳐', '매', '회',
              '2', '##편', '##씩', '총', '8', '##편', '##이', '공개', '##될', '예정이다', '.']
    label_sequence = ['O', 'O', 'O', 'O', 'B-DAT', 'I-DAT', 'B-DUR', 'I-DUR', 'O', 'B-NOH', 'I-NOH',
                      'O', 'O', 'O', 'O', 'B-NOH', 'I-NOH', 'O', 'O', 'B-NOH', 'I-NOH', 'O', 'O', 'O', 'O', 'O']
    """
    prefix_sum_of_token_start_index, sum = [], 0
    for i, token in enumerate(tokens):
        if i == 0:
            prefix_sum_of_token_start_index.append(0)
            sum += len(token) - 1
        else:
            prefix_sum_of_token_start_index.append(sum)
            sum += len(token)

    regex_ner = re.compile('<(.+?):[A-Z]{3}>')  # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 인경우
    regex_filter_res = regex_ner.finditer(target_sentence)

    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = []

    count_of_match = 0
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
        ner_text = match_item[1]  # <4일간:DUR> -> 4일간
        start_index = match_item.start() - 6 * count_of_match  # delete previous '<, :, 3 words tag name, >'
        end_index = match_item.end() - 6 - 6 * count_of_match

        list_of_ner_tag.append(ner_tag)
        list_of_ner_text.append(ner_text)
        list_of_tuple_ner_start_end.append((start_index, end_index))
        count_of_match += 1

    label_sequence = []
    entity_index = 0
    is_entity_still_B = True

    for tup in zip(tokens, prefix_sum_of_token_start_index):
        token, index = tup

        if '▁' in token:  # 주의할 점!! '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
            index += 1  # 토큰이 띄어쓰기를 앞단에 포함한 경우 index 한개 앞으로 당김 # ('▁13', 9) -> ('13', 10)

        if entity_index < len(list_of_tuple_ner_start_end):
            start, end = list_of_tuple_ner_start_end[entity_index]

            if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                is_entity_still_B = True
                entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                start, end = list_of_tuple_ner_start_end[entity_index]

            if start <= index and index < end:  # <13일:DAT>까지 -> ('▁13', 10, 'B-DAT') ('일까지', 12, 'I-DAT') 이런 경우가 포함됨, 포함 안시키려면 토큰의 length도 계산해서 제어해야함
                entity_tag = list_of_ner_tag[entity_index]
                if is_entity_still_B is True:
                    entity_tag = 'B-' + entity_tag
                    label_sequence.append(entity_tag)
                    is_entity_still_B = False
                else:
                    entity_tag = 'I-' + entity_tag
                    label_sequence.append(entity_tag)
            else:
                is_entity_still_B = True
                entity_tag = 'O'
                label_sequence.append(entity_tag)
        else:
            entity_tag = 'O'
            label_sequence.append(entity_tag)

    # truncation
    label_sequence = label_sequence[:max_length - 2]

    # add special tokens
    if cls_token_at_end:
        label_sequence = label_sequence + [NER_CLS_TOKEN, NER_SEP_TOKEN]
    else:
        label_sequence = [NER_CLS_TOKEN] + label_sequence + [NER_SEP_TOKEN]

    # padding
    pad_length = max(max_length - len(label_sequence), 0)
    pad_sequence = [NER_PAD_TOKEN] * pad_length
    label_sequence += pad_sequence

    # encoding
    label_ids = [label_map[label] for label in label_sequence]
    return label_ids


def _convert_examples_to_ner_features(
        examples: List[NERExample],
        tokenizer: PreTrainedTokenizer,
        args: TrainArguments,
        label_list: List[str],
        cls_token_at_end: Optional[bool] = False,
    ):
        """
        `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        """
        label_map = {label: i for i, label in enumerate(label_list)}
        id_to_label = {i: label for i, label in enumerate(label_list)}

        features = []
        for example in examples:
            tokens = tokenizer.tokenize(example.text)
            inputs = tokenizer._encode_plus(
                tokens,
                max_length=args.max_seq_length,
                truncation_strategy=TruncationStrategy.LONGEST_FIRST,
                padding_strategy=PaddingStrategy.MAX_LENGTH,
            )
            label_ids = _process_target_sentence(
                tokens=tokens,
                target_sentence=example.label,
                max_length=args.max_seq_length,
                label_map=label_map,
                cls_token_at_end=cls_token_at_end,
            )
            features.append(NERFeatures(**inputs, label_ids=label_ids))

        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("sentence: %s" % (example.text))
            logger.info("target: %s" % (example.label))
            logger.info("tokens: %s" % (" ".join(tokenizer.convert_ids_to_tokens(features[i].input_ids))))
            logger.info("label: %s" % (" ".join([id_to_label[label_id] for label_id in features[i].label_ids])))
            logger.info("features: %s" % features[i])

        return features


class NERDataset(Dataset):

    def __init__(
            self,
            args: TrainArguments,
            tokenizer: PreTrainedTokenizer,
            corpus: NERCorpus,
            mode: Optional[str] = "train",
            convert_examples_to_features_fn=_convert_examples_to_ner_features,
    ):
        if corpus is not None:
            self.corpus = corpus
        else:
            raise KeyError("corpus is not valid")
        if not mode in ["train", "val", "test"]:
            raise KeyError(f"mode({mode}) is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_cache_dir,
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
                examples = self.corpus.get_examples(corpus_fpath, mode)
                self.features = convert_examples_to_features_fn(
                    examples,
                    tokenizer,
                    args,
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

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.corpus.get_labels()
