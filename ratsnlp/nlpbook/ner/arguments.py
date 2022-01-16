import os
from glob import glob
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
        default="/content/Korpora",
        metadata={"help": "The root directory of the downstream data."}
    )
    downstream_model_dir: str = field(
        default="/gdrive/My Drive/nlpbook/checkpoint-ner",
        metadata={"help": "The output model dir."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
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
        default=None,
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
    epochs: int = field(
        default=3,
        metadata={"help": "max epochs"}
    )
    batch_size: int = field(
        default=32,
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
    tpu_cores: int = field(
        default=0,
        metadata={"help": "Enable TPU with 1 core or 8 cores"}
    )


@dataclass
class NERDeployArguments:

    def __init__(
            self,
            pretrained_model_name=None,
            downstream_model_dir=None,
            downstream_model_checkpoint_fpath=None,
            downstream_model_labelmap_fpath=None,
            max_seq_length=128,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.max_seq_length = max_seq_length
        if downstream_model_checkpoint_fpath is not None and downstream_model_labelmap_fpath is not None:
            self.downstream_model_checkpoint_fpath = downstream_model_checkpoint_fpath
            self.downstream_model_labelmap_fpath = downstream_model_labelmap_fpath
        elif downstream_model_dir is not None:
            ckpt_file_names = glob(os.path.join(downstream_model_dir, "*.ckpt"))
            ckpt_file_names = [el for el in ckpt_file_names if "temp" not in el and "tmp" not in el]
            if len(ckpt_file_names) == 0:
                raise Exception(f"downstream_model_dir \"{downstream_model_dir}\" is not valid")
            selected_fname = ckpt_file_names[-1]
            min_val_loss = os.path.split(selected_fname)[-1].replace(".ckpt", "").split("=")[-1].split("-")[0]
            try:
                for ckpt_file_name in ckpt_file_names:
                    val_loss = os.path.split(ckpt_file_name)[-1].replace(".ckpt", "").split("=")[-1].split("-")[0]
                    if float(val_loss) < float(min_val_loss):
                        selected_fname = ckpt_file_name
                        min_val_loss = val_loss
            except:
                raise Exception(f"the ckpt file name of downstream_model_directory \"{downstream_model_dir}\" is not valid")
            self.downstream_model_checkpoint_fpath = selected_fname
            self.downstream_model_labelmap_fpath = os.path.join(downstream_model_dir, "label_map.txt")
        else:
            raise Exception("Either downstream_model_dir or downstream_model_checkpoint_fpath must be entered.")
        print(f"downstream_model_checkpoint_fpath: {self.downstream_model_checkpoint_fpath}")
        print(f"downstream_model_labelmap_fpath: {self.downstream_model_labelmap_fpath}")
