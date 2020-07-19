import numpy as np
from typing import Callable, Dict
from sklearn.metrics import f1_score
from transformers import EvalPrediction


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if "classification" in task_name:
            preds = np.argmax(p.predictions, axis=1)
            return acc_and_f1(preds, p.label_ids)
        else:
            return {"none": 0}
    return compute_metrics_fn