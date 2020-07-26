import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from transformers import TrainingArguments as FineTuningArguments


def load_arguments(json_file_path=None):
    parser = HfArgumentParser((PretrainedModelArguments, DownstreamDataArguments, FineTuningArguments))
    if json_file_path is not None:
        pretrained_model_args, downstream_data_args, fine_tuning_args = parser.parse_json_file(json_file=json_file_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        pretrained_model_args, downstream_data_args, fine_tuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        pretrained_model_args, downstream_data_args, fine_tuning_args = parser.parse_args_into_dataclasses()
    return pretrained_model_args, downstream_data_args, fine_tuning_args


@dataclass
class PretrainedModelArguments:

    pretrained_model_name: str = field(
        default="kobert", metadata={"help": "pretrained model name"}
    )
    pretrained_model_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from web"}
    )
    loss_type: str = field(
        default="classification", metadata={"help": "loss type (classification or regression)"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # output_dir: Optional[str] = field(
    #     default=None, metadata={"help": "Where do you want to store the fine tuned model"}
    # )


@dataclass
class DownstreamDataArguments:

    corpus_name: str = field(metadata={"help": "The name of the downstream data."})
    task_name: str = field(metadata={"help": "The name of the downstream task."})
    corpus_dir: str = field(
        metadata={"help": "The input corpus dir. Should contain the .txt files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the cached data"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()
        self.corpus_name = self.corpus_name.lower()

