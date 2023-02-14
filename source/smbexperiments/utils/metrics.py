import torch
from torch.utils.data import DataLoader

from smbexperiments.utils import use_GPU



##############################################################################
### METRICS (taken from https://github.com/IssamLaradji/sls/blob/master/src/metrics.py)
##############################################################################


def softmax_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc


def compute_loss(model, dataset):
    metric_function = softmax_loss

    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=1024)

    score_sum = 0.
    for images, labels in loader:
        if use_GPU:
            images, labels = images.cuda(), labels.cuda()

        score_sum += metric_function(model, images, labels).item() * images.shape[0]

    score = float(score_sum / len(loader.dataset))

    return score


def compute_accuracy(model, dataset):
    metric_function = softmax_accuracy

    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=1024)

    score_sum = 0.
    for images, labels in loader:
        if use_GPU:
            images, labels = images.cuda(), labels.cuda()

        score_sum += metric_function(model, images, labels).item() * images.shape[0]

    score = float(score_sum / len(loader.dataset))

    return score