import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)


def load_arguments(json_file_path=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if json_file_path is not None:
        model_args, data_args, training_args = parser.parse_json_file(json_file=json_file_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


@dataclass
class ModelArguments:
    """
    huggingface transformers
    """

    # huggingface 용도
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # ratsnlp 용도
    pretrained_model_name: str = field(
        default="kobert", metadata={"help": "pretrained model name"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    """
    huggingface transformers
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    data_name: str = field(metadata={"help": "The name of the data."})
    task_name: str = field(metadata={"help": "The name of the task."})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()

