"""Started from ignite code for EpochMetric
https://pytorch.org/ignite/_modules/ignite/metrics/epoch_metric.html#EpochMetric
"""

import warnings
import numpy as np
import torch
from ignite.metrics import EpochMetric


class FairEpochMetric(EpochMetric):
    """Class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output
    datatype should be `float32`. Target datatype should be `long`.

    .. warning::

        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.


    - `update` must receive output of the form `(y_pred, y)`.

    If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`

    Args:
        compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
            `predictions` and `targets` and returns a scalar.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """

    def __init__(self, compute_fn, output_transform=lambda x: x):
        super(FairEpochMetric, self).__init__(compute_fn, output_transform=output_transform)

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.long)
        self._sensattr = torch.tensor([], dtype=torch.long)

    def update(self, output):
        y_pred, y, a = output['ypred'], output['y'], output['a']

        if y_pred.ndimension() not in (1, 2):
            raise ValueError("Predictions should be of shape (batch_size, n_classes) or (batch_size, )")

        if y.ndimension() not in (1, 2):
            raise ValueError("Targets should be of shape (batch_size, n_classes) or (batch_size, )")

        if a.ndimension() not in (1, 2):
            raise ValueError("Sensitive attributes should be of shape (batch_size, n_classes) or (batch_size, )")

        if y.ndimension() == 2:
            if not torch.equal(y ** 2, y):
                raise ValueError('Targets should be binary (0 or 1)')

        if a.ndimension() == 2:
            if not torch.equal(a ** 2, a):
                raise ValueError('Sensitive attributes should be binary (0 or 1)')

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        if a.ndimension() == 2 and a.shape[1] == 1:
            a = a.squeeze(dim=-1)

        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)
        a = a.type_as(self._sensattr)

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)
        self._sensattr = torch.cat([self._sensattr, a], dim=0)

        # Check once the signature and execution of compute_fn
        if self._predictions.shape == y_pred.shape:
            try:
                self.compute_fn(self._predictions, self._targets, self._sensattr)
            except Exception as e:
                warnings.warn("Probably, there can be a problem with `compute_fn`:\n {}".format(e),
                              RuntimeWarning)

    def compute(self):
        return self.compute_fn(self._predictions, self._targets, self._sensattr)


def delta_dp_compute_fn(yhat, y, a):
    y = y.numpy()
    yhat = yhat.numpy()
    a = a.numpy()

    avg_yhat_a0 = np.sum(np.multiply(yhat, 1 - a)) / np.sum(1 - a)
    avg_yhat_a1 = np.sum(np.multiply(yhat, a)) / np.sum(a)
    return abs(avg_yhat_a0 - avg_yhat_a1)

class DeltaDP(FairEpochMetric):
    def __init__(self):
        super(DeltaDP, self).__init__(delta_dp_compute_fn)




def accuracy_compute_fn(yhat, y, a):
    y = y.numpy()
    yhat = yhat.numpy()
    err = np.abs(y - yhat).mean()
    return 1. - err


class Accuracy(FairEpochMetric):
    def __init__(self):
        super(Accuracy, self).__init__(accuracy_compute_fn)




