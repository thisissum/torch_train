import torch
from torch import nn
import os
import prettytable
from tqdm import tqdm

class Callback(object):
    """Base class of callbacks
    """

    def __init__(self, name=None):
        self.name = name
        self.to_register = {}

    def register_to(self, context):
        context.update(self.to_register)

    def get_name(self):
        return self.name

    def on_train_start(self, context):
        pass

    def on_epoch_start(self, context):
        pass

    def on_batch_start(self, context):
        pass

    def on_batch_end(self, context):
        pass

    def on_epoch_end(self, context):
        pass

    def on_train_end(self, context):
        pass

class CallbackList(Callback):
    """
    Callback list to contain callback classes
    Use for-loop to call every callback at each part of train loop
    """
    def __init__(self, callbacks):
        super(CallbackList, self).__init__()
        self.callbacks = callbacks

    def _check(self):
        to_pop = None
        for i, callback in enumerate(self.callbacks):
            if not isinstance(callback, Callback): # if callbacks in trainer.build is None, use empty callback
                self.callbacks[i] = Callback(str(callback))
            if callback.get_name() == 'display': # move display callback to the last one
                to_pop = i
        if to_pop is not None:
            self.callbacks.append(self.callbacks.pop(to_pop))

    def on_train_start(self, context):
        for callback in self.callbacks:
            callback.register_to(context)
            callback.on_train_start(context)

    def on_epoch_start(self, context):
        for callback in self.callbacks:
            callback.on_epoch_start(context)

    def on_batch_start(self, context):
        for callback in self.callbacks:
            callback.on_batch_start(context)

    def on_train_end(self, context):
        for callback in self.callbacks:
            callback.on_train_end(context)

    def on_epoch_end(self, context):
        for callback in self.callbacks:
            callback.on_epoch_end(context)

    def on_batch_end(self, context):
        for callback in self.callbacks:
            callback.on_batch_end(context)


class EarlyStopingCallback(Callback):
    """Stop training when metric on validation set not imporved
    Args:
        mode: str, 'min' or 'max', metric used to define 'improved'
        delta: float, if new metric is in delta range of old metric, no improved
        patience: int, times to wait for improved
    """
    def __init__(self, metric_name, mode='max', delta=0, patience=5, path='./checkpoint/last_best.pt'):
        super(EarlyStopingCallback, self).__init__(name='early_stopping')
        self.metric_name = metric_name
        self.mode = mode
        self.delta = delta
        self.path = path
        self.ori_patience = patience
        self.cur_patience = patience


        self.last_best = None
        self.sign = 1 if mode == 'max' else -1

        self.to_register = {'early_stop': False, 'ckpt_path':path, 'save_model':False}


    def on_epoch_end(self, context):
        if context[self.metric_name]:
            value = context[self.metric_name]
            # when first call on_epoch_end
            if self.last_best is None:
                self.last_best = value
                context.update({'save_model': True})
                return

            if self.sign * (value - self.last_best) > self.delta:
                # better than before
                self.cur_patience = self.ori_patience
                context.update({'save_model':True})
                # torch.save(self.model.state_dict(), self.path)
                self.last_best = value
            else:
                self.cur_patience -= 1
                context.update({'save_model':False})
                if self.cur_patience <= 0:
                    context.update({'early_stop':True})
        else:
            return

class CheckpointCallback(Callback):
    def __init__(self, from_epoch=0, dir='./checkpoint'):
        super(CheckpointCallback, self).__init__(name='checkpoint')
        self.from_epoch = from_epoch
        self.dir = dir

        self.cur_epoch = 0
        self.to_register({'save_model': False, 'ckpt_path':''})

    def on_epoch_start(self, context):
        self.cur_epoch += 1

    def on_epoch_end(self, context):
        if self.cur_epoch >= self.from_epoch:
            ckpt_path = os.path.join(self.dir, 'epoch_{}_weight.pt'.format(self.cur_epoch))
            context.update({'ckpt_path':ckpt_path})


class DisplayCallback(Callback):
    def __init__(self, to_file=None):
        super(DisplayCallback, self).__init__(name='display')
        self.cur_epoch = 0
        self.to_file = to_file
        self.table = prettytable.PrettyTable()

    def on_train_start(self, context):
        self.cur_epoch += 1

    def on_epoch_end(self, context):
        self.table.add_column('epoch', self.cur_epoch)
        for field in context.get_fields():
            self.table.add_column(field, context[field])
        tqdm.write(self.table)
        if self.to_file:
            with open(self.to_file, 'w', encoding='utf-8') as f:
                f.write(str(self.table))
        self.table = prettytable.PrettyTable()

