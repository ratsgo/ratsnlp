import os
from dataclasses import dataclass, field


@dataclass
class NERTrainArguments:

    pretrained_model_name: str = field(
        default="beomi/kcbert-base",
        metadata={"help": "pretrained model name"}
    )
    downstream_task_name: str = field(
        default="named-entity-recognition",
        metadata={"help": "The name of the downstream data."}
    )
    downstream_corpus_name: str = field(
        default="ner",
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
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
    overwrite_model: bool = field(
        default=False,
        metadata={"help": "overwrite model checkpoints."}
    )
    save_top_k: int = field(
        default=1,
        metadata={"help": "save top k model checkpoints."}
    )
    monitor: str = field(
        default="max val_acc",
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
        default=5e-6,
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
    cpu_workers: int = field(
        default=os.cpu_count(),
        metadata={"help": "number of CPU workers"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Enable train on FP16"}
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "do train or not"}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "do validation or not"}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "do test or not"}
    )
    tpu_cores: int = field(
        default=0,
        metadata={"help": "Enable TPU with 1 core or 8 cores"}
    )
    report_cycle: int = field(
        default=100,
        metadata={"help": "Report (Train Metrics) Cycle"}
    )
    stat_window_length: int = field(
        default=30,
        metadata={"help": "train statistics window length"}
    )


@dataclass
class NERDeployArguments:

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
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
