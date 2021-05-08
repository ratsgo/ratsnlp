import os
from dataclasses import dataclass, field


@dataclass
class QATrainArguments:

    pretrained_model_name: str = field(
        default="beomi/kcbert-base",
        metadata={"help": "pretrained model name"}
    )
    downstream_corpus_name: str = field(
        default=None,
        metadata={"help": "The name of the downstream data."}
    )
    downstream_corpus_root_dir: str = field(
        default="/root/Korpora",
        metadata={"help": "The root directory of the downstream data."}
    )
    downstream_model_dir: str = field(
        default=None,
        metadata={"help": "The output model dir."}
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        }
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will "
                    "be truncated to this length."
        }
    )
    threads: int = field(
        default=4,
        metadata={
            "help": "the number of threads, using for preprocessing"
        }
    )
    cpu_workers: int = field(
        default=os.cpu_count(),
        metadata={"help": "number of CPU workers"}
    )
    save_top_k: int = field(
        default=1,
        metadata={"help": "save top k model checkpoints."}
    )
    monitor: str = field(
        default="min val_loss",
        metadata={"help": "monitor condition (save top k)"}
    )
    seed: int = field(
        default=7,
        metadata={"help": "random seed."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    force_download: bool = field(
        default=False,
        metadata={"help": "force to download downstream data and pretrained models."}
    )
    test_mode: bool = field(
        default=False,
        metadata={"help": "Test Mode enables `fast_dev_run`"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "learning rate"}
    )
    optimizer: str = field(
        default="AdamW",
        metadata={"help": "optimizer"}
    )
    lr_scheduler: str = field(
        default="exp",
        metadata={"help": "ExponentialLR or CosineAnnealingWarmRestarts"}
    )
    epochs: int = field(
        default=20,
        metadata={"help": "max epochs"}
    )
    batch_size: int = field(
        default=0,
        metadata={"help": "batch size. if 0, Let PyTorch Lightening find the best batch size"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Enable train on FP16"}
    )
    tpu_cores: int = field(
        default=0,
        metadata={"help": "Enable TPU with 1 core or 8 cores"}
    )
    tqdm_enabled: bool = field(
        default=True,
        metadata={"help": "do tqdn enabled or not"}
    )


@dataclass
class QADeployArguments:

    pretrained_model_name: str = field(
        default=None,
        metadata={"help": "The name of the pretrained model"}
    )
    downstream_model_checkpoint_path: str = field(
        default=None,
        metadata={"help": "The output model checkpoint path."}
    )
    downstream_model_labelmap_path: str = field(
        default=None,
        metadata={"help": "The output model label map path."}
    )
    downstream_task_name: str = field(
        default=None,
        metadata={"help": "The name of the downstream task."}
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will "
                    "be truncated to this length."
        }
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        }
    )
