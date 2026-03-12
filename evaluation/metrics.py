import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def anomaly_score(input_img, recon_img):

    error = F.mse_loss(
        recon_img,
        input_img,
        reduction="none"
    )

    error = error.mean(dim=[1,2,3])

    return error


def compute_auroc(labels, scores):

    labels = torch.tensor(labels).numpy()
    scores = torch.tensor(scores).numpy()

    return roc_auc_score(labels, scores)