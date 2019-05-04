"""Helper models (not including VAEs)."""
import torch
from torch import nn

def calculate_delta_dp(yhat, a):
    avg_yhat_a0 = torch.mean(yhat[1 - a].float())
    avg_yhat_a1 = torch.mean(yhat[a].float())
    delta_dp = abs(avg_yhat_a0 - avg_yhat_a1)
    if not torch.isnan(delta_dp):
        return delta_dp
    else:
        return avg_yhat_a0 if not torch.isnan(avg_yhat_a0) else avg_yhat_a1


class MLPClassifier(torch.nn.Module): 
    def __init__(self, n, h, c):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(n, h),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(h, int(h / 2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(int(h / 2), c)
                )

    def forward(self, x, y, a):
        logits = self.model(x)
        logprobs = nn.LogSoftmax(dim=1)(logits)

        criterion = torch.nn.NLLLoss()
        loss = criterion(logprobs, y)

        _, predicted_classes = torch.max(logprobs, 1)
        accuracy = (predicted_classes == y).float().mean()
        delta_dp = calculate_delta_dp(predicted_classes, a)

        probs = torch.exp(logprobs)
        avg_pred = torch.mean(probs, dim=0)[1]

        metrics = {'loss': loss,
                   'acc': accuracy,
                   'est_delta_dp': delta_dp,
                   'avg_pred': avg_pred,
                   'avg_class': torch.mean(predicted_classes.float()),
                   'ypred': predicted_classes,
                   'y': y,
                   'a': a
                  }

        return loss, logprobs, metrics


