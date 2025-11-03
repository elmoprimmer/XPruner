import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Accuracy(object):
    def __init__(self):
        self.correct = None
        self.total = None
        self.reset()

        self.predictions = []
        self.targets = []

    def reset(self):
        self.correct = 0
        self.total = 0

        self.predictions = []
        self.targets = []

    @torch.no_grad()
    def update(self, outs, ys):
        if isinstance(outs, (tuple,list)):
            outs = outs[0]
        preds = torch.argmax(outs, dim=1)
        correct_mask = (preds == ys)

        self.correct += correct_mask.sum().item()
        self.total += ys.size(0)

        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(ys.cpu().numpy())

    def compute(self):
        results = {
            "n_correct": self.correct,
            "n_total": self.total,
            "total_accuracy": self.correct / self.total if self.total else 0,
        }
        results["confusion_matrix"] = confusion_matrix(self.targets, self.predictions)

        return results


def evaluate(model, dataloader, metric, device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for imgs, ys in tqdm(dataloader, desc="Evaluating", unit="batch"):
            imgs, ys = imgs.to(device), ys.to(device)
            outs = model(imgs)
            metric.update(outs, ys)

    return metric.compute()

def evaluate_wrapped(model, dataloader, metric, device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for imgs, ys in tqdm(dataloader, desc="Evaluating", unit="batch"):
            imgs, ys = imgs.to(device), ys.to(device)
            outs = model(imgs, y=ys)
            metric.update(outs, ys)

    return metric.compute()


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
