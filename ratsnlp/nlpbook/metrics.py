import torch


def accuracy(preds, labels, ignore_index=None):
    with torch.no_grad():
        assert preds.shape[0] == len(labels)
        correct = torch.sum(preds == labels)
        total = torch.sum(torch.ones_like(labels))
        if ignore_index is not None:
            # 모델이 맞춘 것 가운데 ignore index에 해당하는 것 제외
            correct -= torch.sum(torch.logical_and(preds == ignore_index, preds == labels))
            # accuracy의 분모 가운데 ignore index에 해당하는 것 제외
            total -= torch.sum(labels == ignore_index)
    return correct.to(dtype=torch.float) / total.to(dtype=torch.float)
