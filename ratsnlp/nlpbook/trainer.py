import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


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
        default_root_dir=args.downstream_model_dir,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
        # For TPU Setup
        tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    return checkpoint_callback, trainer
