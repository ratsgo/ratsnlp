import torch


def accuracy(preds, labels, ignore_index=None):
    with torch.no_grad():
        assert preds.shape[0] == len(labels)
        correct = torch.sum(preds == labels)
        total = torch.sum(torch.ones_like(labels))
        if ignore_index is not None:
            correct -= torch.sum(preds == ignore_index)
            total -= torch.sum(labels == ignore_index)
    return correct / total
