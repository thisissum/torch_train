import torch
from torch import nn
import numpy as np
import random
from tqdm import tqdm
from collections import deque
from .callbacks import CallbackList
from .context import Context

# TODO: distributed(): set to multi-gpu
# TODO: 解决callbacks和metrics输入是None和输入不是list的情况
# TODO: 更改trainer中依赖于context信号进行消息传递的架构，至少避免其出现在主fit循环里
# TODO: 加入可以每次都保存模型权重的checkpoint回调函数 Done!

class Trainer(object):
    """
    A tool used for training model in keras way
    Args:
        device: str or torch.device, default None, if None, gpu will be used if exist else cpu
        verbose: bool, default True, add tqdm progressbar to dataloader
        name: str, default 'trainer',name of class
    """
    def __init__(self, device=None, verbose=True, name='trainer'):
        self.device = device if device else \
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.name = name
        self.verbose = verbose
        self.setup = False

    def build(self, model, optimizer, criterion, callbacks=None, metrics=None):
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = CallbackList(callbacks)
        self.callbacks._check()
        self.metrics = MetricList(metrics) if isinstance(metrics, (list, tuple)) else MetricList([metrics])
        self.setup = True

    def _check(self):
        assert self.setup == True, 'You should build trainer before fit'

    def fit(self, train_loader, num_epoch, validation_loader=None):
        """
        Train model with train_loader given
        Args:
            train_loader: torch.utils.data.Dataloader
            num_epoch: int
            validation_loader: torch.utils.data.Dataloader
        """
        # check everything
        self._check()
        # set verbose using tqdm
        if verbose:
            train_loader = tqdm(train_loader)
            validation_loader = tqdm(validation_loader) if validation_loader else None

        # train loop
        context = Context()
        self.callbacks.on_train_start(context)
        for epoch in range(num_epoch):
            self.metrics.clear()
            self.callbacks.on_epoch_start(context)
            for i, data in enumerate(train_loader):
                self.callbacks.on_batch_start(context)
                train_info = self._train_step(data)
                context.update(train_info)
                self.callbacks.on_batch_end(context)

            # add validation step
            if validation_loader:
                self.model.eval()  # change behaivor of batchnorm and dropout
                with torch.no_grad():
                    for i, data in enumerate(validation_loader):
                        validation_info = self._validation_step(data)
                        context.update(validation_info)

                    metric_info = self.metrics.compute_metric()
                    context.update(metric_info)
                self.model.train() # back to train mode

            self.callbacks.on_epoch_end(context)

            
            # check save_model
            if context['save_model']:
                # load best model to trainer
                ckpt_path = context['ckpt_path']
                torch.save(self.model.state_dict(), ckpt_path)
            if context['early_stop']:
                ckpt_path = context['ckpt_path']
                self.model.load_state_dict(torch.load(ckpt_path))
                break

        self.callbacks.on_train_end(context)

    def _train_step(self, data):
        # forward batch
        data = move_to_device(data)
        y_pred, y_true = self.forward_batch(data)
        # compute loss
        loss = self.criterion(y_pred, y_true)
        # update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # summary info in train step
        info = {'train_loss': loss.item()}
        return info

    def predict(self, test_loader):
        output = []
        with torch.no_grad():
            for data in test_loader:
                data = move_to_device(data)
                y_out = self.predict_batch(data)
                output.append(y_out)
        return torch.stack(output)

    def _validation_step(self, data):
        data = move_to_device(data)
        y_pred, y_true = self.forward_batch(data)
        loss = self.criterion(y_pred, y_true)
        self.metrics.update_state(y_pred, y_true)
        info = {'validation_loss': loss.item()}
        return info

    def forward_batch(self, data):
        raise NotImplementedError

    def predict_batch(self, data):
        raise NotImplementedError


