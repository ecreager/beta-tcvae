from numbers import Number
import math
import torch
import os


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)
    # delete old models from more than 10 epochs ago
    n_checkpts_to_save = 10
    old_checkpt = os.path.join(save, 
            'checkpt-%04d.pth' % (epoch - n_checkpts_to_save))
    if os.path.exists(old_checkpt):
        os.remove(old_checkpt)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.97):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def isnan(tensor):
    return (tensor != tensor)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def logit(value, eps=1e-9):
    """https://en.wikipedia.org/wiki/Logit"""
    with torch.no_grad():
        if (value -  torch.clamp(value, 0.0, 1.0)).norm().item() > eps:
            raise ValueError('invalid input value for logit function: {}'.format(value))
    log_odd_pos = (value + eps).log()
    log_odd_neg = (1 - value + eps).log()
    log_odds = log_odd_pos - log_odd_neg
    return log_odds
