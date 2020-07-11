import torch
from torch import nn
import abc
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Metric(object):
    """
    Base class of Metric
    Overwrite function: compute_metric, return metric of updated samples
    Overwrite function: update_state, save listed data into pred_record and true_record
    """

    def __init__(self, name=None, is_validation_metric=False):
        self.pred_record = list()
        self.true_record = list()
        self.cur_metric = None
        self.name = name
        self.is_validation_metric = is_validation_metric

    def get_name(self):
        return self.name

    def set_to_validation(self):
        self.is_validation_metric = True

    def update_state(self, y_pred, y_true):
        pass

    def compute_metric(self):
        pass

    def result(self):
        self.cur_metric = self.compute_metric()
        return self.cur_metric

    def display(self):
        cur_metric = self.result()
        print("Validation Metric {}: {}".format(self.name, cur_metric))

    def get_cur_metric(self):
        return self.cur_metric

    def clear(self):
        self.pred_record = list()
        self.true_record = list()


class MetricList(Metric):
    def __init__(self, metrics):
        super(MetricList, self).__init__(name='metric_list')
        self.metrics = metrics if isinstance(metrics, (list, tuple)) else list(metrics)
        self._check()
        for m in metrics:
            if m.is_validation_metric:
                self.validation_metric = m.name
        
    def _check(self):
        # if all metric is not validation metric, raise error
        # if validation_metric num >= 2, raise error
        validation_metric_num = 0
        for m in self.metrics:
            if m.is_validation_metric == True:
                validation_metric_num += 1
            if validation_metric_num >= 2:
                raise "Validation metric num over 1"
        if validation_metric_num == 0:
            self.metrics[0].set_to_validation()

    def update_state(self, y_pred, y_true):
        for metric in self.metrics:
            metric.update_state(y_pred, y_true)

    def compute_metric(self):
        metrics_dict = dict()
        for metric in self.metrics:
            metrics_dict[metric.name] = metric.result()
        return metrics_dict

    def clear(self):
        for metric in self.metrics:
            metric.clear()


class AccuracyMetric(Metric):
    """
    Compute accuracy of all sample updated in
    """

    def __init__(self, normalize=True, sample_weight=None):
        super(AccuracyMetric, self).__init__(name='accuracy')
        self.normalize = normalize
        self.sample_weight = sample_weight

    def update_state(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=-1).cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred
        self.true_record += y_true

    def compute_metric(self):
        score = accuracy_score(
            self.true_record,
            self.pred_record,
            normalize=self.normalize,
            sample_weight=self.sample_weight
        )
        return score


class F1Metric(Metric):
    """
    Compute F1 score of all sample updated in
    """

    def __init__(self, labels=None, pos_label=1, average='binary', sample_weight=None):
        super(F1Metric, self).__init__(name='{} f1'.format(average))
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight

    def update_state(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=-1).cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred
        self.true_record += y_true

    def compute_metric(self):
        score = f1_score(
            self.true_record,
            self.pred_record,
            labels=self.labels,
            pos_label=self.pos_label,
            average=self.average,
            sample_weight=self.sample_weight
        )
        return score
