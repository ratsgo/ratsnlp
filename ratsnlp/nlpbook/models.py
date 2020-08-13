import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


def get_tokenizer(args):
    return AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name
        else args.pretrained_model_cache_dir,
        do_lower_case=False,
        cache_dir=args.downstream_model_dir,
    )


def get_pretrained_model(args, num_labels):
    pretrained_model_config = AutoConfig.from_pretrained(
        args.pretrained_model_config_name if args.pretrained_model_config_name
        else args.pretrained_model_cache_dir,
        num_labels=num_labels,
        finetuning_task=args.downstream_task_name,
        cache_dir=args.downstream_model_dir,
    )
    if args.downstream_task_name == "document-classification":
        return AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_model_cache_dir,
            from_tf=bool(".ckpt" in args.pretrained_model_cache_dir),
            config=pretrained_model_config,
            cache_dir=args.downstream_model_dir,
        )
    elif args.downstream_task_name == "":
        return None
    else:
        return None


def get_trainer(args):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.abspath(args.downstream_model_dir),
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
        prefix='',
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        checkpoint_callback=checkpoint_callback,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
        # For TPU Setup
        tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    return checkpoint_callback, trainer
