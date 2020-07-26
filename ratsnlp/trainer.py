import os
import torch
import logging
import numpy as np
from typing import Callable, Dict, Optional
from .metrics import build_compute_metrics_fn
from transformers import TrainingArguments as FineTuningArguments
from transformers import PreTrainedModel, EvalPrediction
from transformers import Trainer as BaseTrainer


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):

    def __init__(self,
                 pretrained_model: PreTrainedModel,
                 tune_args: FineTuningArguments,
                 train_dataset: Optional[torch.utils.data.dataset] = None,
                 eval_dataset: Optional[torch.utils.data.dataset] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        ):
        super().__init__(
            model=pretrained_model,
            args=tune_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

    def fit(self):
        # Training
        if self.args.do_train:
            self.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            self.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if self.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)

        # Evaluation
        eval_results = {}
        if self.args.do_eval:
            logger.info("*** Evaluate ***")
            eval_datasets = [self.eval_dataset]
            for eval_dataset in eval_datasets:
                self.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
                eval_result = self.evaluate(eval_dataset=eval_dataset)
                output_eval_file = os.path.join(
                    training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
                )
                if trainer.is_world_master():
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))
                eval_results.update(eval_result)

        if self.args.do_predict:
            logging.info("*** Test ***")
            test_datasets = [self.test_dataset]
            for test_dataset in test_datasets:
                predictions = self.predict(test_dataset=test_dataset).predictions
                if "classification" in test_dataset.args.task_name:
                    predictions = np.argmax(predictions, axis=1)
                output_test_file = os.path.join(
                    self.args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
                )
                if self.is_world_master():
                    with open(output_test_file, "w") as writer:
                        logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if "regression" in test_dataset.args.task_name:
                                writer.write("%d\t%3.3f\n" % (index, item))
                            else:
                                item = test_dataset.get_labels()[item]
                                writer.write("%d\t%s\n" % (index, item))
